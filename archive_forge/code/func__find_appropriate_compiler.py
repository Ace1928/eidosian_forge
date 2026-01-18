import os
import re
import sys
def _find_appropriate_compiler(_config_vars):
    """Find appropriate C compiler for extension module builds"""
    if 'CC' in os.environ:
        return _config_vars
    cc = oldcc = _config_vars['CC'].split()[0]
    if not _find_executable(cc):
        cc = _find_build_tool('clang')
    elif os.path.basename(cc).startswith('gcc'):
        data = _read_output("'%s' --version" % (cc.replace("'", '\'"\'"\''),))
        if data and 'llvm-gcc' in data:
            cc = _find_build_tool('clang')
    if not cc:
        raise SystemError('Cannot locate working compiler')
    if cc != oldcc:
        for cv in _COMPILER_CONFIG_VARS:
            if cv in _config_vars and cv not in os.environ:
                cv_split = _config_vars[cv].split()
                cv_split[0] = cc if cv != 'CXX' else cc + '++'
                _save_modified_value(_config_vars, cv, ' '.join(cv_split))
    return _config_vars