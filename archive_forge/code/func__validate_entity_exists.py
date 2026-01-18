from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_entity_exists(self, _list, var_file, key, setup):
    isValid = True
    key_setup = self._key_setup(setup, key)
    _mapping = var_file.get(key)
    if _mapping is None:
        return isValid
    for x in _mapping:
        if key_setup not in x.keys():
            print("%s%sdictionary key '%s' is not included in %s[%s].%s" % (FAIL, PREFIX, key_setup, key, x.keys(), END))
            isValid = False
        if isValid and x[key_setup] not in _list:
            print("%s%s%s entity '%s':'%s' does not exist in the setup.\n%sThe entities which exists in the setup are: %s.%s" % (FAIL, PREFIX, key, key_setup, x[key_setup], PREFIX, _list, END))
            isValid = False
    if isValid:
        print("%s%sFinished validation for '%s' for key name '%s' with success.%s" % (INFO, PREFIX, key, key_setup, END))
    return isValid