from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
class BtrfsSubvolume(object):
    """
    Wrapper class providing convenience methods for inspection of a btrfs subvolume
    """

    def __init__(self, filesystem, subvolume_id):
        self.__filesystem = filesystem
        self.__subvolume_id = subvolume_id

    def get_filesystem(self):
        return self.__filesystem

    def is_mounted(self):
        mountpoints = self.get_mountpoints()
        return mountpoints is not None and len(mountpoints) > 0

    def is_filesystem_root(self):
        return 5 == self.__subvolume_id

    def is_filesystem_default(self):
        return self.__filesystem.default_subvolid == self.__subvolume_id

    def get_mounted_path(self):
        mountpoints = self.get_mountpoints()
        if mountpoints is not None and len(mountpoints) > 0:
            return mountpoints[0]
        elif self.parent is not None:
            parent = self.__filesystem.get_subvolume_by_id(self.parent)
            parent_path = parent.get_mounted_path()
            if parent_path is not None:
                return parent_path + os.path.sep + self.name
        else:
            return None

    def get_mountpoints(self):
        return self.__filesystem.get_mountpoints_by_subvolume_id(self.__subvolume_id)

    def get_child_relative_path(self, absolute_child_path):
        """
        Get the relative path from this subvolume to the named child subvolume.
        The provided parameter is expected to be normalized as by normalize_subvolume_path.
        """
        path = self.path
        if absolute_child_path.startswith(path):
            relative = absolute_child_path[len(path):]
            return re.sub('^/*', '', relative)
        else:
            raise BtrfsModuleException("Path '%s' doesn't start with '%s'" % (absolute_child_path, path))

    def get_parent_subvolume(self):
        parent_id = self.parent
        return self.__filesystem.get_subvolume_by_id(parent_id) if parent_id is not None else None

    def get_child_subvolumes(self):
        return self.__filesystem.get_subvolume_children(self.__subvolume_id)

    @property
    def __info(self):
        return self.__filesystem.get_subvolume_info_for_id(self.__subvolume_id)

    @property
    def id(self):
        return self.__subvolume_id

    @property
    def name(self):
        return self.path.split('/').pop()

    @property
    def path(self):
        return self.__info['path']

    @property
    def parent(self):
        return self.__info['parent']