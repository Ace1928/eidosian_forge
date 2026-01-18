import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _DefinedSymroots(self, target):
    config_list = target.GetProperty('buildConfigurationList')
    symroots = set()
    for config in config_list.GetProperty('buildConfigurations'):
        setting = config.GetProperty('buildSettings')
        if 'SYMROOT' in setting:
            symroots.add(setting['SYMROOT'])
        else:
            symroots.add(None)
    if len(symroots) == 1 and None in symroots:
        return set()
    return symroots