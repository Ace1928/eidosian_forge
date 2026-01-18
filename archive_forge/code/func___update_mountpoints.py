from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __update_mountpoints(self, mountpoints):
    self.__mountpoints = dict()
    for i in mountpoints:
        subvolid = i['subvolid']
        mountpoint = i['mountpoint']
        if subvolid not in self.__mountpoints:
            self.__mountpoints[subvolid] = []
        self.__mountpoints[subvolid].append(mountpoint)