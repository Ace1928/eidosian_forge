from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
class BtrfsFilesystemsProvider(object):
    """
    Provides methods to query available btrfs filesystems
    """

    def __init__(self, module):
        self.__module = module
        self.__provider = BtrfsInfoProvider(module)
        self.__filesystems = None

    def get_matching_filesystem(self, criteria):
        if criteria['device'] is not None:
            criteria['device'] = os.path.realpath(criteria['device'])
        self.__check_init()
        matching = [f for f in self.__filesystems.values() if self.__filesystem_matches_criteria(f, criteria)]
        if len(matching) == 1:
            return matching[0]
        else:
            raise BtrfsModuleException('Found %d filesystems matching criteria uuid=%s label=%s device=%s' % (len(matching), criteria['uuid'], criteria['label'], criteria['device']))

    def __filesystem_matches_criteria(self, filesystem, criteria):
        return (criteria['uuid'] is None or filesystem.uuid == criteria['uuid']) and (criteria['label'] is None or filesystem.label == criteria['label']) and (criteria['device'] is None or filesystem.contains_device(criteria['device']))

    def get_filesystem_for_device(self, device):
        real_device = os.path.realpath(device)
        self.__check_init()
        for fs in self.__filesystems.values():
            if fs.contains_device(real_device):
                return fs
        return None

    def get_filesystems(self):
        self.__check_init()
        return list(self.__filesystems.values())

    def __check_init(self):
        if self.__filesystems is None:
            self.__filesystems = dict()
            for f in self.__provider.get_filesystems():
                uuid = f['uuid']
                self.__filesystems[uuid] = BtrfsFilesystem(f, self.__provider, self.__module)