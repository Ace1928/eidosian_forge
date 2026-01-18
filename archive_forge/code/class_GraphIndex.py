import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class GraphIndex:
    """An index for data with embedded graphs.

    The index maps keys to a list of key reference lists, and a value.
    Each node has the same number of key reference lists. Each key reference
    list can be empty or an arbitrary length. The value is an opaque NULL
    terminated string without any newlines. The storage of the index is
    hidden in the interface: keys and key references are always tuples of
    bytestrings, never the internal representation (e.g. dictionary offsets).

    It is presumed that the index will not be mutated - it is static data.

    Successive iter_all_entries calls will read the entire index each time.
    Additionally, iter_entries calls will read the index linearly until the
    desired keys are found. XXX: This must be fixed before the index is
    suitable for production use. :XXX
    """

    def __init__(self, transport, name, size, unlimited_cache=False, offset=0):
        """Open an index called name on transport.

        :param transport: A breezy.transport.Transport.
        :param name: A path to provide to transport API calls.
        :param size: The size of the index in bytes. This is used for bisection
            logic to perform partial index reads. While the size could be
            obtained by statting the file this introduced an additional round
            trip as well as requiring stat'able transports, both of which are
            avoided by having it supplied. If size is None, then bisection
            support will be disabled and accessing the index will just stream
            all the data.
        :param offset: Instead of starting the index data at offset 0, start it
            at an arbitrary offset.
        """
        self._transport = transport
        self._name = name
        self._bisect_nodes = None
        self._nodes = None
        self._parsed_byte_map = []
        self._parsed_key_map = []
        self._key_count = None
        self._keys_by_offset = None
        self._nodes_by_key = None
        self._size = size
        self._bytes_read = 0
        self._base_offset = offset

    def __eq__(self, other):
        """Equal when self and other were created with the same parameters."""
        return isinstance(self, type(other)) and self._transport == other._transport and (self._name == other._name) and (self._size == other._size)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, GraphIndex) and (not isinstance(other, InMemoryGraphIndex)):
            raise TypeError(other)
        return hash(self) < hash(other)

    def __hash__(self):
        return hash((type(self), self._transport, self._name, self._size))

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._transport.abspath(self._name))

    def _buffer_all(self, stream=None):
        """Buffer all the index data.

        Mutates self._nodes and self.keys_by_offset.
        """
        if self._nodes is not None:
            return
        if 'index' in debug.debug_flags:
            trace.mutter('Reading entire index %s', self._transport.abspath(self._name))
        if stream is None:
            stream = self._transport.get(self._name)
            if self._base_offset != 0:
                stream = BytesIO(stream.read()[self._base_offset:])
        try:
            self._read_prefix(stream)
            self._expected_elements = 3 + self._key_length
            line_count = 0
            self._keys_by_offset = {}
            self._nodes = {}
            self._nodes_by_key = None
            trailers = 0
            pos = stream.tell()
            lines = stream.read().split(b'\n')
        finally:
            stream.close()
        del lines[-1]
        _, _, _, trailers = self._parse_lines(lines, pos)
        for key, absent, references, value in self._keys_by_offset.values():
            if absent:
                continue
            if self.node_ref_lists:
                node_value = (value, self._resolve_references(references))
            else:
                node_value = value
            self._nodes[key] = node_value
        if trailers != 1:
            raise BadIndexData(self)

    def clear_cache(self):
        """Clear out any cached/memoized values.

        This can be called at any time, but generally it is used when we have
        extracted some information, but don't expect to be requesting any more
        from this index.
        """

    def external_references(self, ref_list_num):
        """Return references that are not present in this index.
        """
        self._buffer_all()
        if ref_list_num + 1 > self.node_ref_lists:
            raise ValueError('No ref list %d, index has %d ref lists' % (ref_list_num, self.node_ref_lists))
        refs = set()
        nodes = self._nodes
        for key, (value, ref_lists) in nodes.items():
            ref_list = ref_lists[ref_list_num]
            refs.update([ref for ref in ref_list if ref not in nodes])
        return refs

    def _get_nodes_by_key(self):
        if self._nodes_by_key is None:
            nodes_by_key = {}
            if self.node_ref_lists:
                for key, (value, references) in self._nodes.items():
                    key_dict = nodes_by_key
                    for subkey in key[:-1]:
                        key_dict = key_dict.setdefault(subkey, {})
                    key_dict[key[-1]] = (key, value, references)
            else:
                for key, value in self._nodes.items():
                    key_dict = nodes_by_key
                    for subkey in key[:-1]:
                        key_dict = key_dict.setdefault(subkey, {})
                    key_dict[key[-1]] = (key, value)
            self._nodes_by_key = nodes_by_key
        return self._nodes_by_key

    def iter_all_entries(self):
        """Iterate over all keys within the index.

        :return: An iterable of (index, key, value) or (index, key, value, reference_lists).
            The former tuple is used when there are no reference lists in the
            index, making the API compatible with simple key:value index types.
            There is no defined order for the result iteration - it will be in
            the most efficient order for the index.
        """
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(3, 'iter_all_entries scales with size of history.')
        if self._nodes is None:
            self._buffer_all()
        if self.node_ref_lists:
            for key, (value, node_ref_lists) in self._nodes.items():
                yield (self, key, value, node_ref_lists)
        else:
            for key, value in self._nodes.items():
                yield (self, key, value)

    def _read_prefix(self, stream):
        signature = stream.read(len(self._signature()))
        if not signature == self._signature():
            raise BadIndexFormatSignature(self._name, GraphIndex)
        options_line = stream.readline()
        if not options_line.startswith(_OPTION_NODE_REFS):
            raise BadIndexOptions(self)
        try:
            self.node_ref_lists = int(options_line[len(_OPTION_NODE_REFS):-1])
        except ValueError:
            raise BadIndexOptions(self)
        options_line = stream.readline()
        if not options_line.startswith(_OPTION_KEY_ELEMENTS):
            raise BadIndexOptions(self)
        try:
            self._key_length = int(options_line[len(_OPTION_KEY_ELEMENTS):-1])
        except ValueError:
            raise BadIndexOptions(self)
        options_line = stream.readline()
        if not options_line.startswith(_OPTION_LEN):
            raise BadIndexOptions(self)
        try:
            self._key_count = int(options_line[len(_OPTION_LEN):-1])
        except ValueError:
            raise BadIndexOptions(self)

    def _resolve_references(self, references):
        """Return the resolved key references for references.

        References are resolved by looking up the location of the key in the
        _keys_by_offset map and substituting the key name, preserving ordering.

        :param references: An iterable of iterables of key locations. e.g.
            [[123, 456], [123]]
        :return: A tuple of tuples of keys.
        """
        node_refs = []
        for ref_list in references:
            node_refs.append(tuple([self._keys_by_offset[ref][0] for ref in ref_list]))
        return tuple(node_refs)

    @staticmethod
    def _find_index(range_map, key):
        """Helper for the _parsed_*_index calls.

        Given a range map - [(start, end), ...], finds the index of the range
        in the map for key if it is in the map, and if it is not there, the
        immediately preceeding range in the map.
        """
        result = bisect_right(range_map, key) - 1
        if result + 1 < len(range_map):
            if range_map[result + 1][0] == key[0]:
                return result + 1
        return result

    def _parsed_byte_index(self, offset):
        """Return the index of the entry immediately before offset.

        e.g. if the parsed map has regions 0,10 and 11,12 parsed, meaning that
        there is one unparsed byte (the 11th, addressed as[10]). then:
        asking for 0 will return 0
        asking for 10 will return 0
        asking for 11 will return 1
        asking for 12 will return 1
        """
        key = (offset, 0)
        return self._find_index(self._parsed_byte_map, key)

    def _parsed_key_index(self, key):
        """Return the index of the entry immediately before key.

        e.g. if the parsed map has regions (None, 'a') and ('b','c') parsed,
        meaning that keys from None to 'a' inclusive, and 'b' to 'c' inclusive
        have been parsed, then:
        asking for '' will return 0
        asking for 'a' will return 0
        asking for 'b' will return 1
        asking for 'e' will return 1
        """
        search_key = (key, b'')
        return self._find_index(self._parsed_key_map, search_key)

    def _is_parsed(self, offset):
        """Returns True if offset has been parsed."""
        index = self._parsed_byte_index(offset)
        if index == len(self._parsed_byte_map):
            return offset < self._parsed_byte_map[index - 1][1]
        start, end = self._parsed_byte_map[index]
        return offset >= start and offset < end

    def _iter_entries_from_total_buffer(self, keys):
        """Iterate over keys when the entire index is parsed."""
        nodes = self._nodes
        keys = [key for key in keys if key in nodes]
        if self.node_ref_lists:
            for key in keys:
                value, node_refs = nodes[key]
                yield (self, key, value, node_refs)
        else:
            for key in keys:
                yield (self, key, nodes[key])

    def iter_entries(self, keys):
        """Iterate over keys within the index.

        :param keys: An iterable providing the keys to be retrieved.
        :return: An iterable as per iter_all_entries, but restricted to the
            keys supplied. No additional keys will be returned, and every
            key supplied that is in the index will be returned.
        """
        keys = set(keys)
        if not keys:
            return []
        if self._size is None and self._nodes is None:
            self._buffer_all()
        if self._nodes is None and len(keys) * 20 > self.key_count():
            self._buffer_all()
        if self._nodes is not None:
            return self._iter_entries_from_total_buffer(keys)
        else:
            return (result[1] for result in bisect_multi.bisect_multi_bytes(self._lookup_keys_via_location, self._size, keys))

    def iter_entries_prefix(self, keys):
        """Iterate over keys within the index using prefix matching.

        Prefix matching is applied within the tuple of a key, not to within
        the bytestring of each key element. e.g. if you have the keys ('foo',
        'bar'), ('foobar', 'gam') and do a prefix search for ('foo', None) then
        only the former key is returned.

        WARNING: Note that this method currently causes a full index parse
        unconditionally (which is reasonably appropriate as it is a means for
        thunking many small indices into one larger one and still supplies
        iter_all_entries at the thunk layer).

        :param keys: An iterable providing the key prefixes to be retrieved.
            Each key prefix takes the form of a tuple the length of a key, but
            with the last N elements 'None' rather than a regular bytestring.
            The first element cannot be 'None'.
        :return: An iterable as per iter_all_entries, but restricted to the
            keys with a matching prefix to those supplied. No additional keys
            will be returned, and every match that is in the index will be
            returned.
        """
        keys = set(keys)
        if not keys:
            return
        if self._nodes is None:
            self._buffer_all()
        if self._key_length == 1:
            for key in keys:
                _sanity_check_key(self, key)
                if self.node_ref_lists:
                    value, node_refs = self._nodes[key]
                    yield (self, key, value, node_refs)
                else:
                    yield (self, key, self._nodes[key])
            return
        nodes_by_key = self._get_nodes_by_key()
        yield from _iter_entries_prefix(self, nodes_by_key, keys)

    def _find_ancestors(self, keys, ref_list_num, parent_map, missing_keys):
        """See BTreeIndex._find_ancestors."""
        found_keys = set()
        search_keys = set()
        for index, key, value, refs in self.iter_entries(keys):
            parent_keys = refs[ref_list_num]
            found_keys.add(key)
            parent_map[key] = parent_keys
            search_keys.update(parent_keys)
        missing_keys.update(set(keys).difference(found_keys))
        search_keys = search_keys.difference(parent_map)
        return search_keys

    def key_count(self):
        """Return an estimate of the number of keys in this index.

        For GraphIndex the estimate is exact.
        """
        if self._key_count is None:
            self._read_and_parse([_HEADER_READV])
        return self._key_count

    def _lookup_keys_via_location(self, location_keys):
        """Public interface for implementing bisection.

        If _buffer_all has been called, then all the data for the index is in
        memory, and this method should not be called, as it uses a separate
        cache because it cannot pre-resolve all indices, which buffer_all does
        for performance.

        :param location_keys: A list of location(byte offset), key tuples.
        :return: A list of (location_key, result) tuples as expected by
            breezy.bisect_multi.bisect_multi_bytes.
        """
        readv_ranges = []
        for location, key in location_keys:
            if self._bisect_nodes and key in self._bisect_nodes:
                continue
            index = self._parsed_key_index(key)
            if len(self._parsed_key_map) and self._parsed_key_map[index][0] <= key and (self._parsed_key_map[index][1] >= key or self._parsed_byte_map[index][1] == self._size):
                continue
            index = self._parsed_byte_index(location)
            if len(self._parsed_byte_map) and self._parsed_byte_map[index][0] <= location and (self._parsed_byte_map[index][1] > location):
                continue
            length = 800
            if location + length > self._size:
                length = self._size - location
            if length > 0:
                readv_ranges.append((location, length))
        if self._bisect_nodes is None:
            readv_ranges.append(_HEADER_READV)
        self._read_and_parse(readv_ranges)
        result = []
        if self._nodes is not None:
            for location, key in location_keys:
                if key not in self._nodes:
                    result.append(((location, key), False))
                elif self.node_ref_lists:
                    value, refs = self._nodes[key]
                    result.append(((location, key), (self, key, value, refs)))
                else:
                    result.append(((location, key), (self, key, self._nodes[key])))
            return result
        pending_references = []
        pending_locations = set()
        for location, key in location_keys:
            if key in self._bisect_nodes:
                if self.node_ref_lists:
                    value, refs = self._bisect_nodes[key]
                    wanted_locations = []
                    for ref_list in refs:
                        for ref in ref_list:
                            if ref not in self._keys_by_offset:
                                wanted_locations.append(ref)
                    if wanted_locations:
                        pending_locations.update(wanted_locations)
                        pending_references.append((location, key))
                        continue
                    result.append(((location, key), (self, key, value, self._resolve_references(refs))))
                else:
                    result.append(((location, key), (self, key, self._bisect_nodes[key])))
                continue
            else:
                index = self._parsed_key_index(key)
                if self._parsed_key_map[index][0] <= key and (self._parsed_key_map[index][1] >= key or self._parsed_byte_map[index][1] == self._size):
                    result.append(((location, key), False))
                    continue
            index = self._parsed_byte_index(location)
            if key < self._parsed_key_map[index][0]:
                direction = -1
            else:
                direction = +1
            result.append(((location, key), direction))
        readv_ranges = []
        for location in pending_locations:
            length = 800
            if location + length > self._size:
                length = self._size - location
            if length > 0:
                readv_ranges.append((location, length))
        self._read_and_parse(readv_ranges)
        if self._nodes is not None:
            for location, key in pending_references:
                value, refs = self._nodes[key]
                result.append(((location, key), (self, key, value, refs)))
            return result
        for location, key in pending_references:
            value, refs = self._bisect_nodes[key]
            result.append(((location, key), (self, key, value, self._resolve_references(refs))))
        return result

    def _parse_header_from_bytes(self, bytes):
        """Parse the header from a region of bytes.

        :param bytes: The data to parse.
        :return: An offset, data tuple such as readv yields, for the unparsed
            data. (which may length 0).
        """
        signature = bytes[0:len(self._signature())]
        if not signature == self._signature():
            raise BadIndexFormatSignature(self._name, GraphIndex)
        lines = bytes[len(self._signature()):].splitlines()
        options_line = lines[0]
        if not options_line.startswith(_OPTION_NODE_REFS):
            raise BadIndexOptions(self)
        try:
            self.node_ref_lists = int(options_line[len(_OPTION_NODE_REFS):])
        except ValueError:
            raise BadIndexOptions(self)
        options_line = lines[1]
        if not options_line.startswith(_OPTION_KEY_ELEMENTS):
            raise BadIndexOptions(self)
        try:
            self._key_length = int(options_line[len(_OPTION_KEY_ELEMENTS):])
        except ValueError:
            raise BadIndexOptions(self)
        options_line = lines[2]
        if not options_line.startswith(_OPTION_LEN):
            raise BadIndexOptions(self)
        try:
            self._key_count = int(options_line[len(_OPTION_LEN):])
        except ValueError:
            raise BadIndexOptions(self)
        header_end = len(signature) + len(lines[0]) + len(lines[1]) + len(lines[2]) + 3
        self._parsed_bytes(0, (), header_end, ())
        self._expected_elements = 3 + self._key_length
        self._keys_by_offset = {}
        self._bisect_nodes = {}
        return (header_end, bytes[header_end:])

    def _parse_region(self, offset, data):
        """Parse node data returned from a readv operation.

        :param offset: The byte offset the data starts at.
        :param data: The data to parse.
        """
        end = offset + len(data)
        high_parsed = offset
        while True:
            index = self._parsed_byte_index(high_parsed)
            if end < self._parsed_byte_map[index][1]:
                return
            high_parsed, last_segment = self._parse_segment(offset, data, end, index)
            if last_segment:
                return

    def _parse_segment(self, offset, data, end, index):
        """Parse one segment of data.

        :param offset: Where 'data' begins in the file.
        :param data: Some data to parse a segment of.
        :param end: Where data ends
        :param index: The current index into the parsed bytes map.
        :return: True if the parsed segment is the last possible one in the
            range of data.
        :return: high_parsed_byte, last_segment.
            high_parsed_byte is the location of the highest parsed byte in this
            segment, last_segment is True if the parsed segment is the last
            possible one in the data block.
        """
        trim_end = None
        if offset < self._parsed_byte_map[index][1]:
            trim_start = self._parsed_byte_map[index][1] - offset
            start_adjacent = True
        elif offset == self._parsed_byte_map[index][1]:
            trim_start = None
            start_adjacent = True
        else:
            trim_start = None
            start_adjacent = False
        if end == self._size:
            trim_end = None
            end_adjacent = True
            last_segment = True
        elif index + 1 == len(self._parsed_byte_map):
            trim_end = None
            end_adjacent = False
            last_segment = True
        elif end == self._parsed_byte_map[index + 1][0]:
            trim_end = None
            end_adjacent = True
            last_segment = True
        elif end > self._parsed_byte_map[index + 1][0]:
            trim_end = self._parsed_byte_map[index + 1][0] - offset
            end_adjacent = True
            last_segment = end < self._parsed_byte_map[index + 1][1]
        else:
            trim_end = None
            end_adjacent = False
            last_segment = True
        if not start_adjacent:
            if trim_start is None:
                trim_start = data.find(b'\n') + 1
            else:
                trim_start = data.find(b'\n', trim_start) + 1
            if not trim_start != 0:
                raise AssertionError('no \n was present')
        if not end_adjacent:
            if trim_end is None:
                trim_end = data.rfind(b'\n') + 1
            else:
                trim_end = data.rfind(b'\n', None, trim_end) + 1
            if not trim_end != 0:
                raise AssertionError('no \n was present')
        trimmed_data = data[trim_start:trim_end]
        if not trimmed_data:
            raise AssertionError('read unneeded data [%d:%d] from [%d:%d]' % (trim_start, trim_end, offset, offset + len(data)))
        if trim_start:
            offset += trim_start
        lines = trimmed_data.split(b'\n')
        del lines[-1]
        pos = offset
        first_key, last_key, nodes, _ = self._parse_lines(lines, pos)
        for key, value in nodes:
            self._bisect_nodes[key] = value
        self._parsed_bytes(offset, first_key, offset + len(trimmed_data), last_key)
        return (offset + len(trimmed_data), last_segment)

    def _parse_lines(self, lines, pos):
        key = None
        first_key = None
        trailers = 0
        nodes = []
        for line in lines:
            if line == b'':
                if self._size:
                    if not self._size == pos + 1:
                        raise AssertionError('{} {}'.format(self._size, pos))
                trailers += 1
                continue
            elements = line.split(b'\x00')
            if len(elements) != self._expected_elements:
                raise BadIndexData(self)
            key = tuple([element for element in elements[:self._key_length]])
            if first_key is None:
                first_key = key
            absent, references, value = elements[-3:]
            ref_lists = []
            for ref_string in references.split(b'\t'):
                ref_lists.append(tuple([int(ref) for ref in ref_string.split(b'\r') if ref]))
            ref_lists = tuple(ref_lists)
            self._keys_by_offset[pos] = (key, absent, ref_lists, value)
            pos += len(line) + 1
            if absent:
                continue
            if self.node_ref_lists:
                node_value = (value, ref_lists)
            else:
                node_value = value
            nodes.append((key, node_value))
        return (first_key, key, nodes, trailers)

    def _parsed_bytes(self, start, start_key, end, end_key):
        """Mark the bytes from start to end as parsed.

        Calling self._parsed_bytes(1,2) will mark one byte (the one at offset
        1) as parsed.

        :param start: The start of the parsed region.
        :param end: The end of the parsed region.
        """
        index = self._parsed_byte_index(start)
        new_value = (start, end)
        new_key = (start_key, end_key)
        if index == -1:
            self._parsed_byte_map.insert(index, new_value)
            self._parsed_key_map.insert(index, new_key)
            return
        if index + 1 < len(self._parsed_byte_map) and self._parsed_byte_map[index][1] == start and (self._parsed_byte_map[index + 1][0] == end):
            self._parsed_byte_map[index] = (self._parsed_byte_map[index][0], self._parsed_byte_map[index + 1][1])
            self._parsed_key_map[index] = (self._parsed_key_map[index][0], self._parsed_key_map[index + 1][1])
            del self._parsed_byte_map[index + 1]
            del self._parsed_key_map[index + 1]
        elif self._parsed_byte_map[index][1] == start:
            self._parsed_byte_map[index] = (self._parsed_byte_map[index][0], end)
            self._parsed_key_map[index] = (self._parsed_key_map[index][0], end_key)
        elif index + 1 < len(self._parsed_byte_map) and self._parsed_byte_map[index + 1][0] == end:
            self._parsed_byte_map[index + 1] = (start, self._parsed_byte_map[index + 1][1])
            self._parsed_key_map[index + 1] = (start_key, self._parsed_key_map[index + 1][1])
        else:
            self._parsed_byte_map.insert(index + 1, new_value)
            self._parsed_key_map.insert(index + 1, new_key)

    def _read_and_parse(self, readv_ranges):
        """Read the ranges and parse the resulting data.

        :param readv_ranges: A prepared readv range list.
        """
        if not readv_ranges:
            return
        if self._nodes is None and self._bytes_read * 2 >= self._size:
            self._buffer_all()
            return
        base_offset = self._base_offset
        if base_offset != 0:
            readv_ranges = [(start + base_offset, size) for start, size in readv_ranges]
        readv_data = self._transport.readv(self._name, readv_ranges, True, self._size + self._base_offset)
        for offset, data in readv_data:
            offset -= base_offset
            self._bytes_read += len(data)
            if offset < 0:
                data = data[-offset:]
                offset = 0
            if offset == 0 and len(data) == self._size:
                self._buffer_all(BytesIO(data))
                return
            if self._bisect_nodes is None:
                if not offset == 0:
                    raise AssertionError()
                offset, data = self._parse_header_from_bytes(data)
            self._parse_region(offset, data)

    def _signature(self):
        """The file signature for this index type."""
        return _SIGNATURE

    def validate(self):
        """Validate that everything in the index can be accessed."""
        for node in self.iter_all_entries():
            pass