from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class XrefTable:

    def __init__(self):
        self.existing_entries = {}
        self.new_entries = {}
        self.deleted_entries = {0: 65536}
        self.reading_finished = False

    def __setitem__(self, key, value):
        if self.reading_finished:
            self.new_entries[key] = value
        else:
            self.existing_entries[key] = value
        if key in self.deleted_entries:
            del self.deleted_entries[key]

    def __getitem__(self, key):
        try:
            return self.new_entries[key]
        except KeyError:
            return self.existing_entries[key]

    def __delitem__(self, key):
        if key in self.new_entries:
            generation = self.new_entries[key][1] + 1
            del self.new_entries[key]
            self.deleted_entries[key] = generation
        elif key in self.existing_entries:
            generation = self.existing_entries[key][1] + 1
            self.deleted_entries[key] = generation
        elif key in self.deleted_entries:
            generation = self.deleted_entries[key]
        else:
            msg = 'object ID ' + str(key) + " cannot be deleted because it doesn't exist"
            raise IndexError(msg)

    def __contains__(self, key):
        return key in self.existing_entries or key in self.new_entries

    def __len__(self):
        return len(set(self.existing_entries.keys()) | set(self.new_entries.keys()) | set(self.deleted_entries.keys()))

    def keys(self):
        return set(self.existing_entries.keys()) - set(self.deleted_entries.keys()) | set(self.new_entries.keys())

    def write(self, f):
        keys = sorted(set(self.new_entries.keys()) | set(self.deleted_entries.keys()))
        deleted_keys = sorted(set(self.deleted_entries.keys()))
        startxref = f.tell()
        f.write(b'xref\n')
        while keys:
            prev = None
            for index, key in enumerate(keys):
                if prev is None or prev + 1 == key:
                    prev = key
                else:
                    contiguous_keys = keys[:index]
                    keys = keys[index:]
                    break
            else:
                contiguous_keys = keys
                keys = None
            f.write(b'%d %d\n' % (contiguous_keys[0], len(contiguous_keys)))
            for object_id in contiguous_keys:
                if object_id in self.new_entries:
                    f.write(b'%010d %05d n \n' % self.new_entries[object_id])
                else:
                    this_deleted_object_id = deleted_keys.pop(0)
                    check_format_condition(object_id == this_deleted_object_id, f'expected the next deleted object ID to be {object_id}, instead found {this_deleted_object_id}')
                    try:
                        next_in_linked_list = deleted_keys[0]
                    except IndexError:
                        next_in_linked_list = 0
                    f.write(b'%010d %05d f \n' % (next_in_linked_list, self.deleted_entries[object_id]))
        return startxref