import os
import re
import sys
def _remove_unsupported_archs(_config_vars):
    """Remove any unsupported archs from config vars"""
    if 'CC' in os.environ:
        return _config_vars
    if re.search('-arch\\s+ppc', _config_vars['CFLAGS']) is not None:
        status = os.system("echo 'int main{};' | '%s' -c -arch ppc -x c -o /dev/null /dev/null 2>/dev/null" % (_config_vars['CC'].replace("'", '\'"\'"\''),))
        if status:
            for cv in _UNIVERSAL_CONFIG_VARS:
                if cv in _config_vars and cv not in os.environ:
                    flags = _config_vars[cv]
                    flags = re.sub('-arch\\s+ppc\\w*\\s', ' ', flags)
                    _save_modified_value(_config_vars, cv, flags)
    return _config_vars