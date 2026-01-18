from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __parse_mountpoint_pairs(self, line):
    pattern = re.compile('^TARGET="(?P<target>.*)"\\s+SOURCE="(?P<source>.*)"\\s+FSTYPE="(?P<fstype>.*)"\\s+OPTIONS="(?P<options>.*)"\\s*$')
    match = pattern.search(line)
    if match is not None:
        groups = match.groupdict()
        return {'mountpoint': groups['target'], 'device': groups['source'], 'subvolid': self.__extract_mount_subvolid(groups['options'])}
    else:
        raise BtrfsModuleException("Failed to parse findmnt result for line: '%s'" % line)