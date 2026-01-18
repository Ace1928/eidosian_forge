from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _get_dup_network(self, var_file):
    _return_set = set()
    _mapping = var_file.get(self.network_map)
    if _mapping is None or len(_mapping) < 1:
        print('%s%sNetwork has not been initialized in var file%s' % (WARN, PREFIX, END))
        return _return_set
    _primary1 = set()
    key1_a = 'primary_profile_name'
    key1_b = 'primary_network_name'
    key1_c = 'primary_network_dc'
    for x in _mapping:
        if x[key1_a] is None or x[key1_b] is None:
            print("%s%sNetwork '%s' is not initialized in map %s %s%s" % (FAIL, PREFIX, x, x[key1_a], x[key1_b], END))
            sys.exit()
        primary_dc_name = ''
        if key1_c in x:
            primary_dc_name = x[key1_c]
        map_key = x[key1_a] + '_' + x[key1_b] + '_' + primary_dc_name
        if map_key in _primary1:
            _return_set.add(map_key)
        else:
            _primary1.add(map_key)
    _second1 = set()
    val1_a = 'secondary_profile_name'
    val1_b = 'secondary_network_name'
    val1_c = 'secondary_network_dc'
    for x in _mapping:
        if x[val1_a] is None or x[val1_b] is None:
            print("%s%sThe following network mapping is not initialized in var file mapping:\n  %s:'%s'\n  %s:'%s'%s" % (FAIL, PREFIX, val1_a, x[val1_a], val1_b, x[val1_b], END))
            sys.exit()
        secondary_dc_name = ''
        if val1_c in x:
            secondary_dc_name = x[val1_c]
        map_key = x[val1_a] + '_' + x[val1_b] + '_' + secondary_dc_name
        if map_key in _second1:
            _return_set.add(map_key)
        else:
            _second1.add(map_key)
    return _return_set