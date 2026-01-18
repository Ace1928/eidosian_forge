from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _preload_cache(self):
    """Make sure all file-ids are in _fileid_to_entry_cache"""
    if self._fully_cached:
        return
    cache = self._fileid_to_entry_cache
    for key, entry in self.id_to_entry.iteritems():
        file_id = key[0]
        if file_id not in cache:
            ie = self._bytes_to_entry(entry)
            cache[file_id] = ie
        else:
            ie = cache[file_id]
    last_parent_id = last_parent_ie = None
    pid_items = self.parent_id_basename_to_file_id.iteritems()
    for key, child_file_id in pid_items:
        if key == (b'', b''):
            if child_file_id != self.root_id:
                raise ValueError('Data inconsistency detected. We expected data with key ("","") to match the root id, but %s != %s' % (child_file_id, self.root_id))
            continue
        parent_id, basename = key
        ie = cache[child_file_id]
        if parent_id == last_parent_id:
            parent_ie = last_parent_ie
        else:
            parent_ie = cache[parent_id]
        if parent_ie.kind != 'directory':
            raise ValueError('Data inconsistency detected. An entry in the parent_id_basename_to_file_id map has parent_id {%s} but the kind of that object is %r not "directory"' % (parent_id, parent_ie.kind))
        if parent_ie._children is None:
            parent_ie._children = {}
        basename = basename.decode('utf-8')
        if basename in parent_ie._children:
            existing_ie = parent_ie._children[basename]
            if existing_ie != ie:
                raise ValueError('Data inconsistency detected. Two entries with basename %r were found in the parent entry {%s}' % (basename, parent_id))
        if basename != ie.name:
            raise ValueError('Data inconsistency detected. In the parent_id_basename_to_file_id map, file_id {%s} is listed as having basename %r, but in the id_to_entry map it is %r' % (child_file_id, basename, ie.name))
        parent_ie._children[basename] = ie
    self._fully_cached = True