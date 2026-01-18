from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_roles(f):
    f.write('\n# Mapping for role\n')
    f.write('# Fill in any roles which should be mapped between sites.\n')
    f.write('dr_role_mappings: \n')
    f.write('- primary_name: \n')
    f.write('  secondary_name: \n\n')