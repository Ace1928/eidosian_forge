import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def SetDestination(self, path):
    """Set the dstSubfolderSpec and dstPath properties from path.

    path may be specified in the same notation used for XCHierarchicalElements,
    specifically, "$(DIR)/path".
    """
    path_tree_match = self.path_tree_re.search(path)
    if path_tree_match:
        path_tree = path_tree_match.group(1)
        if path_tree in self.path_tree_first_to_subfolder:
            subfolder = self.path_tree_first_to_subfolder[path_tree]
            relative_path = path_tree_match.group(3)
            if relative_path is None:
                relative_path = ''
            if subfolder == 16 and path_tree_match.group(4) is not None:
                path_tree = path_tree_match.group(4)
                relative_path = path_tree_match.group(6)
                separator = '/'
                if path_tree in self.path_tree_second_to_subfolder:
                    subfolder = self.path_tree_second_to_subfolder[path_tree]
                    if relative_path is None:
                        relative_path = ''
                        separator = ''
                    if path_tree == 'XPCSERVICES_FOLDER_PATH':
                        relative_path = '$(CONTENTS_FOLDER_PATH)/XPCServices' + separator + relative_path
                else:
                    relative_path = path_tree_match.group(3)
        else:
            subfolder = 0
            relative_path = path
    elif path.startswith('/'):
        subfolder = 0
        relative_path = path[1:]
    else:
        raise ValueError(f"Can't use path {path} in a {self.__class__.__name__}")
    self._properties['dstPath'] = relative_path
    self._properties['dstSubfolderSpec'] = subfolder