import os
import re
import sys
def _check_for_unavailable_sdk(_config_vars):
    """Remove references to any SDKs not available"""
    cflags = _config_vars.get('CFLAGS', '')
    m = re.search('-isysroot\\s*(\\S+)', cflags)
    if m is not None:
        sdk = m.group(1)
        if not os.path.exists(sdk):
            for cv in _UNIVERSAL_CONFIG_VARS:
                if cv in _config_vars and cv not in os.environ:
                    flags = _config_vars[cv]
                    flags = re.sub('-isysroot\\s*\\S+(?:\\s|$)', ' ', flags)
                    _save_modified_value(_config_vars, cv, flags)
    return _config_vars