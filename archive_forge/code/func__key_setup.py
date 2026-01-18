from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _key_setup(self, setup, key):
    if setup == 'primary':
        if key == 'dr_import_storages':
            return 'dr_primary_name'
        if key == 'dr_network_mappings':
            return ['primary_profile_name', 'primary_network_name', 'primary_network_dc']
        return 'primary_name'
    elif setup == 'secondary':
        if key == 'dr_import_storages':
            return 'dr_secondary_name'
        if key == 'dr_network_mappings':
            return ['secondary_profile_name', 'secondary_network_name', 'secondary_network_dc']
        return 'secondary_name'