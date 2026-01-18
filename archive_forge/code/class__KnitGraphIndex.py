import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
class _KnitGraphIndex:
    """A KnitVersionedFiles index layered on GraphIndex."""

    def __init__(self, graph_index, is_locked, deltas=False, parents=True, add_callback=None, track_external_parent_refs=False):
        """Construct a KnitGraphIndex on a graph_index.

        :param graph_index: An implementation of breezy.index.GraphIndex.
        :param is_locked: A callback to check whether the object should answer
            queries.
        :param deltas: Allow delta-compressed records.
        :param parents: If True, record knits parents, if not do not record
            parents.
        :param add_callback: If not None, allow additions to the index and call
            this callback with a list of added GraphIndex nodes:
            [(node, value, node_refs), ...]
        :param is_locked: A callback, returns True if the index is locked and
            thus usable.
        :param track_external_parent_refs: If True, record all external parent
            references parents from added records.  These can be retrieved
            later by calling get_missing_parents().
        """
        self._add_callback = add_callback
        self._graph_index = graph_index
        self._deltas = deltas
        self._parents = parents
        if deltas and (not parents):
            raise KnitCorrupt(self, 'Cannot do delta compression without parent tracking.')
        self.has_graph = parents
        self._is_locked = is_locked
        self._missing_compression_parents = set()
        if track_external_parent_refs:
            self._key_dependencies = _KeyRefs()
        else:
            self._key_dependencies = None

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._graph_index)

    def add_records(self, records, random_id=False, missing_compression_parents=False):
        """Add multiple records to the index.

        This function does not insert data into the Immutable GraphIndex
        backing the KnitGraphIndex, instead it prepares data for insertion by
        the caller and checks that it is safe to insert then calls
        self._add_callback with the prepared GraphIndex nodes.

        :param records: a list of tuples:
                         (key, options, access_memo, parents).
        :param random_id: If True the ids being added were randomly generated
            and no check for existence will be performed.
        :param missing_compression_parents: If True the records being added are
            only compressed against texts already in the index (or inside
            records). If False the records all refer to unavailable texts (or
            texts inside records) as compression parents.
        """
        if not self._add_callback:
            raise errors.ReadOnlyError(self)
        keys = {}
        compression_parents = set()
        key_dependencies = self._key_dependencies
        for key, options, access_memo, parents in records:
            if self._parents:
                parents = tuple(parents)
                if key_dependencies is not None:
                    key_dependencies.add_references(key, parents)
            index, pos, size = access_memo
            if b'no-eol' in options:
                value = b'N'
            else:
                value = b' '
            value += b'%d %d' % (pos, size)
            if not self._deltas:
                if b'line-delta' in options:
                    raise KnitCorrupt(self, 'attempt to add line-delta in non-delta knit')
            if self._parents:
                if self._deltas:
                    if b'line-delta' in options:
                        node_refs = (parents, (parents[0],))
                        if missing_compression_parents:
                            compression_parents.add(parents[0])
                    else:
                        node_refs = (parents, ())
                else:
                    node_refs = (parents,)
            else:
                if parents:
                    raise KnitCorrupt(self, 'attempt to add node with parents in parentless index.')
                node_refs = ()
            keys[key] = (value, node_refs)
        if not random_id:
            present_nodes = self._get_entries(keys)
            for index, key, value, node_refs in present_nodes:
                parents = node_refs[:1]
                passed = static_tuple.as_tuples(keys[key])
                passed_parents = passed[1][:1]
                if value[0:1] != keys[key][0][0:1] or parents != passed_parents:
                    node_refs = static_tuple.as_tuples(node_refs)
                    raise KnitCorrupt(self, 'inconsistent details in add_records: %s %s' % ((value, node_refs), passed))
                del keys[key]
        result = []
        if self._parents:
            for key, (value, node_refs) in keys.items():
                result.append((key, value, node_refs))
        else:
            for key, (value, node_refs) in keys.items():
                result.append((key, value))
        self._add_callback(result)
        if missing_compression_parents:
            compression_parents.difference_update(keys)
            self._missing_compression_parents.update(compression_parents)
        self._missing_compression_parents.difference_update(keys)

    def scan_unvalidated_index(self, graph_index):
        """Inform this _KnitGraphIndex that there is an unvalidated index.

        This allows this _KnitGraphIndex to keep track of any missing
        compression parents we may want to have filled in to make those
        indices valid.

        :param graph_index: A GraphIndex
        """
        if self._deltas:
            new_missing = graph_index.external_references(ref_list_num=1)
            new_missing.difference_update(self.get_parent_map(new_missing))
            self._missing_compression_parents.update(new_missing)
        if self._key_dependencies is not None:
            for node in graph_index.iter_all_entries():
                self._key_dependencies.add_references(node[1], node[3][0])

    def get_missing_compression_parents(self):
        """Return the keys of missing compression parents.

        Missing compression parents occur when a record stream was missing
        basis texts, or a index was scanned that had missing basis texts.
        """
        return frozenset(self._missing_compression_parents)

    def get_missing_parents(self):
        """Return the keys of missing parents."""
        self._key_dependencies.satisfy_refs_for_keys(self.get_parent_map(self._key_dependencies.get_unsatisfied_refs()))
        return frozenset(self._key_dependencies.get_unsatisfied_refs())

    def _check_read(self):
        """raise if reads are not permitted."""
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)

    def _check_write_ok(self):
        """Assert if writes are not permitted."""
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)

    def _compression_parent(self, an_entry):
        compression_parents = an_entry[3][1]
        if not compression_parents:
            return None
        if len(compression_parents) != 1:
            raise AssertionError('Too many compression parents: %r' % compression_parents)
        return compression_parents[0]

    def get_build_details(self, keys):
        """Get the method, index_memo and compression parent for version_ids.

        Ghosts are omitted from the result.

        :param keys: An iterable of keys.
        :return: A dict of key:
            (index_memo, compression_parent, parents, record_details).
            index_memo
                opaque structure to pass to read_records to extract the raw
                data
            compression_parent
                Content that this record is built upon, may be None
            parents
                Logical parents of this node
            record_details
                extra information about the content which needs to be passed to
                Factory.parse_record
        """
        self._check_read()
        result = {}
        entries = self._get_entries(keys, False)
        for entry in entries:
            key = entry[1]
            if not self._parents:
                parents = ()
            else:
                parents = entry[3][0]
            if not self._deltas:
                compression_parent_key = None
            else:
                compression_parent_key = self._compression_parent(entry)
            noeol = entry[2][0:1] == b'N'
            if compression_parent_key:
                method = 'line-delta'
            else:
                method = 'fulltext'
            result[key] = (self._node_to_position(entry), compression_parent_key, parents, (method, noeol))
        return result

    def _get_entries(self, keys, check_present=False):
        """Get the entries for keys.

        :param keys: An iterable of index key tuples.
        """
        keys = set(keys)
        found_keys = set()
        if self._parents:
            for node in self._graph_index.iter_entries(keys):
                yield node
                found_keys.add(node[1])
        else:
            for node in self._graph_index.iter_entries(keys):
                yield (node[0], node[1], node[2], ())
                found_keys.add(node[1])
        if check_present:
            missing_keys = keys.difference(found_keys)
            if missing_keys:
                raise RevisionNotPresent(missing_keys.pop(), self)

    def get_method(self, key):
        """Return compression method of specified key."""
        return self._get_method(self._get_node(key))

    def _get_method(self, node):
        if not self._deltas:
            return 'fulltext'
        if self._compression_parent(node):
            return 'line-delta'
        else:
            return 'fulltext'

    def _get_node(self, key):
        try:
            return list(self._get_entries([key]))[0]
        except IndexError:
            raise RevisionNotPresent(key, self)

    def get_options(self, key):
        """Return a list representing options.

        e.g. ['foo', 'bar']
        """
        node = self._get_node(key)
        options = [self._get_method(node).encode('ascii')]
        if node[2][0:1] == b'N':
            options.append(b'no-eol')
        return options

    def find_ancestry(self, keys):
        """See CombinedGraphIndex.find_ancestry()"""
        return self._graph_index.find_ancestry(keys, 0)

    def get_parent_map(self, keys):
        """Get a map of the parents of keys.

        :param keys: The keys to look up parents for.
        :return: A mapping from keys to parents. Absent keys are absent from
            the mapping.
        """
        self._check_read()
        nodes = self._get_entries(keys)
        result = {}
        if self._parents:
            for node in nodes:
                result[node[1]] = node[3][0]
        else:
            for node in nodes:
                result[node[1]] = None
        return result

    def get_position(self, key):
        """Return details needed to access the version.

        :return: a tuple (index, data position, size) to hand to the access
            logic to get the record.
        """
        node = self._get_node(key)
        return self._node_to_position(node)
    __contains__ = _mod_index._has_key_from_parent_map

    def keys(self):
        """Get all the keys in the collection.

        The keys are not ordered.
        """
        self._check_read()
        return [node[1] for node in self._graph_index.iter_all_entries()]
    missing_keys = _mod_index._missing_keys_from_parent_map

    def _node_to_position(self, node):
        """Convert an index value to position details."""
        bits = node[2][1:].split(b' ')
        return (node[0], int(bits[0]), int(bits[1]))

    def _sort_keys_by_io(self, keys, positions):
        """Figure out an optimal order to read the records for the given keys.

        Sort keys, grouped by index and sorted by position.

        :param keys: A list of keys whose records we want to read. This will be
            sorted 'in-place'.
        :param positions: A dict, such as the one returned by
            _get_components_positions()
        :return: None
        """

        def get_index_memo(key):
            return positions[key][1]
        return keys.sort(key=get_index_memo)
    _get_total_build_size = _get_total_build_size