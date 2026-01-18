import os
import re
import sys
def _remove_original_values(_config_vars):
    """Remove original unmodified values for testing"""
    for k in list(_config_vars):
        if k.startswith(_INITPRE):
            del _config_vars[k]