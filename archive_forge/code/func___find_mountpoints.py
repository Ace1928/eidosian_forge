from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __find_mountpoints(self):
    command = '%s -t btrfs -nvP' % self.__findmnt_path
    result = self.__module.run_command(command)
    mountpoints = []
    if result[0] == 0:
        lines = result[1].splitlines()
        for line in lines:
            mountpoint = self.__parse_mountpoint_pairs(line)
            mountpoints.append(mountpoint)
    return mountpoints