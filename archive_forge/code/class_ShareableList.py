from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
class ShareableList:
    """Pattern for a mutable list-like object shareable via a shared
    memory block.  It differs from the built-in list type in that these
    lists can not change their overall length (i.e. no append, insert,
    etc.)

    Because values are packed into a memoryview as bytes, the struct
    packing format for any storable value must require no more than 8
    characters to describe its format."""
    _types_mapping = {int: 'q', float: 'd', bool: 'xxxxxxx?', str: '%ds', bytes: '%ds', None.__class__: 'xxxxxx?x'}
    _alignment = 8
    _back_transforms_mapping = {0: lambda value: value, 1: lambda value: value.rstrip(b'\x00').decode(_encoding), 2: lambda value: value.rstrip(b'\x00'), 3: lambda _value: None}

    @staticmethod
    def _extract_recreation_code(value):
        """Used in concert with _back_transforms_mapping to convert values
        into the appropriate Python objects when retrieving them from
        the list as well as when storing them."""
        if not isinstance(value, (str, bytes, None.__class__)):
            return 0
        elif isinstance(value, str):
            return 1
        elif isinstance(value, bytes):
            return 2
        else:
            return 3

    def __init__(self, sequence=None, *, name=None):
        if name is None or sequence is not None:
            sequence = sequence or ()
            _formats = [self._types_mapping[type(item)] if not isinstance(item, (str, bytes)) else self._types_mapping[type(item)] % (self._alignment * (len(item) // self._alignment + 1),) for item in sequence]
            self._list_len = len(_formats)
            assert sum((len(fmt) <= 8 for fmt in _formats)) == self._list_len
            offset = 0
            self._allocated_offsets = [0]
            for fmt in _formats:
                offset += self._alignment if fmt[-1] != 's' else int(fmt[:-1])
                self._allocated_offsets.append(offset)
            _recreation_codes = [self._extract_recreation_code(item) for item in sequence]
            requested_size = struct.calcsize('q' + self._format_size_metainfo + ''.join(_formats) + self._format_packing_metainfo + self._format_back_transform_codes)
            self.shm = SharedMemory(name, create=True, size=requested_size)
        else:
            self.shm = SharedMemory(name)
        if sequence is not None:
            _enc = _encoding
            struct.pack_into('q' + self._format_size_metainfo, self.shm.buf, 0, self._list_len, *self._allocated_offsets)
            struct.pack_into(''.join(_formats), self.shm.buf, self._offset_data_start, *(v.encode(_enc) if isinstance(v, str) else v for v in sequence))
            struct.pack_into(self._format_packing_metainfo, self.shm.buf, self._offset_packing_formats, *(v.encode(_enc) for v in _formats))
            struct.pack_into(self._format_back_transform_codes, self.shm.buf, self._offset_back_transform_codes, *_recreation_codes)
        else:
            self._list_len = len(self)
            self._allocated_offsets = list(struct.unpack_from(self._format_size_metainfo, self.shm.buf, 1 * 8))

    def _get_packing_format(self, position):
        """Gets the packing format for a single value stored in the list."""
        position = position if position >= 0 else position + self._list_len
        if position >= self._list_len or self._list_len < 0:
            raise IndexError('Requested position out of range.')
        v = struct.unpack_from('8s', self.shm.buf, self._offset_packing_formats + position * 8)[0]
        fmt = v.rstrip(b'\x00')
        fmt_as_str = fmt.decode(_encoding)
        return fmt_as_str

    def _get_back_transform(self, position):
        """Gets the back transformation function for a single value."""
        if position >= self._list_len or self._list_len < 0:
            raise IndexError('Requested position out of range.')
        transform_code = struct.unpack_from('b', self.shm.buf, self._offset_back_transform_codes + position)[0]
        transform_function = self._back_transforms_mapping[transform_code]
        return transform_function

    def _set_packing_format_and_transform(self, position, fmt_as_str, value):
        """Sets the packing format and back transformation code for a
        single value in the list at the specified position."""
        if position >= self._list_len or self._list_len < 0:
            raise IndexError('Requested position out of range.')
        struct.pack_into('8s', self.shm.buf, self._offset_packing_formats + position * 8, fmt_as_str.encode(_encoding))
        transform_code = self._extract_recreation_code(value)
        struct.pack_into('b', self.shm.buf, self._offset_back_transform_codes + position, transform_code)

    def __getitem__(self, position):
        position = position if position >= 0 else position + self._list_len
        try:
            offset = self._offset_data_start + self._allocated_offsets[position]
            v, = struct.unpack_from(self._get_packing_format(position), self.shm.buf, offset)
        except IndexError:
            raise IndexError('index out of range')
        back_transform = self._get_back_transform(position)
        v = back_transform(v)
        return v

    def __setitem__(self, position, value):
        position = position if position >= 0 else position + self._list_len
        try:
            item_offset = self._allocated_offsets[position]
            offset = self._offset_data_start + item_offset
            current_format = self._get_packing_format(position)
        except IndexError:
            raise IndexError('assignment index out of range')
        if not isinstance(value, (str, bytes)):
            new_format = self._types_mapping[type(value)]
            encoded_value = value
        else:
            allocated_length = self._allocated_offsets[position + 1] - item_offset
            encoded_value = value.encode(_encoding) if isinstance(value, str) else value
            if len(encoded_value) > allocated_length:
                raise ValueError('bytes/str item exceeds available storage')
            if current_format[-1] == 's':
                new_format = current_format
            else:
                new_format = self._types_mapping[str] % (allocated_length,)
        self._set_packing_format_and_transform(position, new_format, value)
        struct.pack_into(new_format, self.shm.buf, offset, encoded_value)

    def __reduce__(self):
        return (partial(self.__class__, name=self.shm.name), ())

    def __len__(self):
        return struct.unpack_from('q', self.shm.buf, 0)[0]

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self)}, name={self.shm.name!r})'

    @property
    def format(self):
        """The struct packing format used by all currently stored items."""
        return ''.join((self._get_packing_format(i) for i in range(self._list_len)))

    @property
    def _format_size_metainfo(self):
        """The struct packing format used for the items' storage offsets."""
        return 'q' * (self._list_len + 1)

    @property
    def _format_packing_metainfo(self):
        """The struct packing format used for the items' packing formats."""
        return '8s' * self._list_len

    @property
    def _format_back_transform_codes(self):
        """The struct packing format used for the items' back transforms."""
        return 'b' * self._list_len

    @property
    def _offset_data_start(self):
        return (self._list_len + 2) * 8

    @property
    def _offset_packing_formats(self):
        return self._offset_data_start + self._allocated_offsets[-1]

    @property
    def _offset_back_transform_codes(self):
        return self._offset_packing_formats + self._list_len * 8

    def count(self, value):
        """L.count(value) -> integer -- return number of occurrences of value."""
        return sum((value == entry for entry in self))

    def index(self, value):
        """L.index(value) -> integer -- return first index of value.
        Raises ValueError if the value is not present."""
        for position, entry in enumerate(self):
            if value == entry:
                return position
        else:
            raise ValueError(f'{value!r} not in this container')
    __class_getitem__ = classmethod(types.GenericAlias)