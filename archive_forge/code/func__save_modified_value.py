import os
import re
import sys
def _save_modified_value(_config_vars, cv, newvalue):
    """Save modified and original unmodified value of configuration var"""
    oldvalue = _config_vars.get(cv, '')
    if oldvalue != newvalue and _INITPRE + cv not in _config_vars:
        _config_vars[_INITPRE + cv] = oldvalue
    _config_vars[cv] = newvalue