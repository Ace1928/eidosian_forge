from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_failback_leftovers(self):
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    with open(self.default_main_file, 'r') as stream:
        try:
            info_dict = yaml.safe_load(stream)
            running_vms_file = info_dict.get(self.running_vms)
            if os.path.isfile(running_vms_file):
                ans = input('%s%sFile with running vms info already exists from previous failback operation. Do you want to delete it (yes,no)?: %s' % (WARN, PREFIX, END))
                ans = ans.lower()
                if ans in valid and valid[ans]:
                    os.remove(running_vms_file)
                    print("%s%sFile '%s' has been deleted successfully%s" % (INFO, PREFIX, running_vms_file, END))
                else:
                    print("%s%sFile '%s' has not been deleted. It will be used in the next failback operation%s" % (INFO, PREFIX, running_vms_file, END))
        except yaml.YAMLError as exc:
            print("%s%syaml file '%s' could not be loaded%s" % (FAIL, PREFIX, self.default_main_file, END))
            print(exc)
            return False
        except OSError as ex:
            print("%s%sFail to validate failback running vms file '%s'%s" % (FAIL, PREFIX, self.default_main_file, END))
            print(ex)
            return False
    return True