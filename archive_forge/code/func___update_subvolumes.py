from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __update_subvolumes(self, subvolumes):
    self.__subvolumes = dict()
    for subvolume in subvolumes:
        self.__subvolumes[subvolume['id']] = subvolume