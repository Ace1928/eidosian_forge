import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def _get_fields_to_entry(self):
    """Get a function which converts entry fields into a entry record.

        This handles size and executable, as well as parent records.

        :return: A function which takes a list of fields, and returns an
            appropriate record for storing in memory.
        """
    num_present_parents = self._num_present_parents()
    if num_present_parents == 0:

        def fields_to_entry_0_parents(fields, _int=int):
            path_name_file_id_key = (fields[0], fields[1], fields[2])
            return (path_name_file_id_key, [(fields[3], fields[4], _int(fields[5]), fields[6] == b'y', fields[7])])
        return fields_to_entry_0_parents
    elif num_present_parents == 1:

        def fields_to_entry_1_parent(fields, _int=int):
            path_name_file_id_key = (fields[0], fields[1], fields[2])
            return (path_name_file_id_key, [(fields[3], fields[4], _int(fields[5]), fields[6] == b'y', fields[7]), (fields[8], fields[9], _int(fields[10]), fields[11] == b'y', fields[12])])
        return fields_to_entry_1_parent
    elif num_present_parents == 2:

        def fields_to_entry_2_parents(fields, _int=int):
            path_name_file_id_key = (fields[0], fields[1], fields[2])
            return (path_name_file_id_key, [(fields[3], fields[4], _int(fields[5]), fields[6] == b'y', fields[7]), (fields[8], fields[9], _int(fields[10]), fields[11] == b'y', fields[12]), (fields[13], fields[14], _int(fields[15]), fields[16] == b'y', fields[17])])
        return fields_to_entry_2_parents
    else:

        def fields_to_entry_n_parents(fields, _int=int):
            path_name_file_id_key = (fields[0], fields[1], fields[2])
            trees = [(fields[cur], fields[cur + 1], _int(fields[cur + 2]), fields[cur + 3] == b'y', fields[cur + 4]) for cur in range(3, len(fields) - 1, 5)]
            return (path_name_file_id_key, trees)
        return fields_to_entry_n_parents