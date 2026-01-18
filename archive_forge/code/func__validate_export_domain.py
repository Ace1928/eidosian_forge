from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_export_domain(self, var_file):
    domains = var_file[self.domain_map]
    for domain in domains:
        domain_type = domain['dr_storage_domain_type']
        if domain_type == 'export':
            print('%s%sExport storage domain is not supported.%s' % (FAIL, PREFIX, END))
            return False
    return True