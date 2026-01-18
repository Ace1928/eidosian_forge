from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
class BtrfsFilesystem(object):
    """
    Wrapper class providing convenience methods for inspection of a btrfs filesystem
    """

    def __init__(self, info, provider, module):
        self.__provider = provider
        self.__uuid = info['uuid']
        self.__label = info['label']
        self.__devices = info['devices']
        self.__default_subvolid = info['default_subvolid'] if 'default_subvolid' in info else None
        self.__update_mountpoints(info['mountpoints'] if 'mountpoints' in info else [])
        self.__update_subvolumes(info['subvolumes'] if 'subvolumes' in info else [])

    @property
    def uuid(self):
        return self.__uuid

    @property
    def label(self):
        return self.__label

    @property
    def default_subvolid(self):
        return self.__default_subvolid

    @property
    def devices(self):
        return list(self.__devices)

    def refresh(self):
        self.refresh_mountpoints()
        self.refresh_subvolumes()
        self.refresh_default_subvolume()

    def refresh_mountpoints(self):
        mountpoints = self.__provider.get_mountpoints(list(self.__devices))
        self.__update_mountpoints(mountpoints)

    def __update_mountpoints(self, mountpoints):
        self.__mountpoints = dict()
        for i in mountpoints:
            subvolid = i['subvolid']
            mountpoint = i['mountpoint']
            if subvolid not in self.__mountpoints:
                self.__mountpoints[subvolid] = []
            self.__mountpoints[subvolid].append(mountpoint)

    def refresh_subvolumes(self):
        filesystem_path = self.get_any_mountpoint()
        if filesystem_path is not None:
            subvolumes = self.__provider.get_subvolumes(filesystem_path)
            self.__update_subvolumes(subvolumes)

    def __update_subvolumes(self, subvolumes):
        self.__subvolumes = dict()
        for subvolume in subvolumes:
            self.__subvolumes[subvolume['id']] = subvolume

    def refresh_default_subvolume(self):
        filesystem_path = self.get_any_mountpoint()
        if filesystem_path is not None:
            self.__default_subvolid = self.__provider.get_default_subvolume_id(filesystem_path)

    def contains_device(self, device):
        return device in self.__devices

    def contains_subvolume(self, subvolume):
        return self.get_subvolume_by_name(subvolume) is not None

    def get_subvolume_by_id(self, subvolume_id):
        return BtrfsSubvolume(self, subvolume_id) if subvolume_id in self.__subvolumes else None

    def get_subvolume_info_for_id(self, subvolume_id):
        return self.__subvolumes[subvolume_id] if subvolume_id in self.__subvolumes else None

    def get_subvolume_by_name(self, subvolume):
        for subvolume_info in self.__subvolumes.values():
            if subvolume_info['path'] == subvolume:
                return BtrfsSubvolume(self, subvolume_info['id'])
        return None

    def get_any_mountpoint(self):
        for subvol_mountpoints in self.__mountpoints.values():
            if len(subvol_mountpoints) > 0:
                return subvol_mountpoints[0]
        return None

    def get_any_mounted_subvolume(self):
        for subvolid, subvol_mountpoints in self.__mountpoints.items():
            if len(subvol_mountpoints) > 0:
                return self.get_subvolume_by_id(subvolid)
        return None

    def get_mountpoints_by_subvolume_id(self, subvolume_id):
        return self.__mountpoints[subvolume_id] if subvolume_id in self.__mountpoints else []

    def get_nearest_subvolume(self, subvolume):
        """Return the identified subvolume if existing, else the closest matching parent"""
        subvolumes_by_path = self.__get_subvolumes_by_path()
        while len(subvolume) > 1:
            if subvolume in subvolumes_by_path:
                return BtrfsSubvolume(self, subvolumes_by_path[subvolume]['id'])
            else:
                subvolume = re.sub('/[^/]+$', '', subvolume)
        return BtrfsSubvolume(self, 5)

    def get_mountpath_as_child(self, subvolume_name):
        """Find a path to the target subvolume through a mounted ancestor"""
        nearest = self.get_nearest_subvolume(subvolume_name)
        if nearest.path == subvolume_name:
            nearest = nearest.get_parent_subvolume()
        if nearest is None or nearest.get_mounted_path() is None:
            raise BtrfsModuleException("Failed to find a path '%s' through a mounted parent subvolume" % subvolume_name)
        else:
            return nearest.get_mounted_path() + os.path.sep + nearest.get_child_relative_path(subvolume_name)

    def get_subvolume_children(self, subvolume_id):
        return [BtrfsSubvolume(self, x['id']) for x in self.__subvolumes.values() if x['parent'] == subvolume_id]

    def __get_subvolumes_by_path(self):
        result = {}
        for s in self.__subvolumes.values():
            path = s['path']
            result[path] = s
        return result

    def is_mounted(self):
        return self.__mountpoints is not None and len(self.__mountpoints) > 0

    def get_summary(self):
        subvolumes = []
        sources = self.__subvolumes.values() if self.__subvolumes is not None else []
        for subvolume in sources:
            id = subvolume['id']
            subvolumes.append({'id': id, 'path': subvolume['path'], 'parent': subvolume['parent'], 'mountpoints': self.get_mountpoints_by_subvolume_id(id)})
        return {'default_subvolume': self.__default_subvolid, 'devices': self.__devices, 'label': self.__label, 'uuid': self.__uuid, 'subvolumes': subvolumes}