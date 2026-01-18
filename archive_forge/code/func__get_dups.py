from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _get_dups(self, var_file, mappings):
    duplicates = {}
    for mapping in mappings:
        _return_set = set()
        _mapping = var_file.get(mapping[1])
        if _mapping is None or len(_mapping) < 1:
            print('%s%smapping %s is empty in var file%s' % (WARN, PREFIX, mapping[1], END))
            duplicates[mapping[0]] = _return_set
            continue
        _primary = set()
        _second = set()
        _return_set.update(set((x[mapping[2]] for x in _mapping if x[mapping[2]] in _primary or _primary.add(x[mapping[2]]))))
        _return_set.update(set((x[mapping[3]] for x in _mapping if x[mapping[3]] in _second or _second.add(x[mapping[3]]))))
        duplicates[mapping[0]] = _return_set
    return duplicates