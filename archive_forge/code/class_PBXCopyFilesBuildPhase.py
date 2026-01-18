import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXCopyFilesBuildPhase(XCBuildPhase):
    _schema = XCBuildPhase._schema.copy()
    _schema.update({'dstPath': [0, str, 0, 1], 'dstSubfolderSpec': [0, int, 0, 1], 'name': [0, str, 0, 0]})
    path_tree_re = re.compile('^\\$\\((.*?)\\)(/(\\$\\((.*?)\\)(/(.*)|)|(.*)|)|)$')
    path_tree_first_to_subfolder = {'BUILT_PRODUCTS_DIR': 16, 'BUILT_FRAMEWORKS_DIR': 10}
    path_tree_second_to_subfolder = {'WRAPPER_NAME': 1, 'EXECUTABLE_FOLDER_PATH': 6, 'UNLOCALIZED_RESOURCES_FOLDER_PATH': 7, 'JAVA_FOLDER_PATH': 15, 'FRAMEWORKS_FOLDER_PATH': 10, 'SHARED_FRAMEWORKS_FOLDER_PATH': 11, 'SHARED_SUPPORT_FOLDER_PATH': 12, 'PLUGINS_FOLDER_PATH': 13, 'XPCSERVICES_FOLDER_PATH': 16}

    def Name(self):
        if 'name' in self._properties:
            return self._properties['name']
        return 'CopyFiles'

    def FileGroup(self, path):
        return self.PBXProjectAncestor().RootGroupForPath(path)

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