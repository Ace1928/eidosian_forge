from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _get_network_dups(self, networks_setup):
    attributes = [attr['profile_name'] + '_' + attr['network_name'] + '_' + attr['network_dc'] for attr in networks_setup]
    dups = [x for n, x in enumerate(attributes) if x in attributes[:n]]
    return dups