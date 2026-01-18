import os
import re
import sys
def _remove_universal_flags(_config_vars):
    """Remove all universal build arguments from config vars"""
    for cv in _UNIVERSAL_CONFIG_VARS:
        if cv in _config_vars and cv not in os.environ:
            flags = _config_vars[cv]
            flags = re.sub('-arch\\s+\\w+\\s', ' ', flags, flags=re.ASCII)
            flags = re.sub('-isysroot\\s*\\S+', ' ', flags)
            _save_modified_value(_config_vars, cv, flags)
    return _config_vars