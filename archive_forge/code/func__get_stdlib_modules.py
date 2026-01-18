import sys
def _get_stdlib_modules():
    if sys.version_info.major == 3:
        if sys.version_info.minor == 8:
            return stdlib3_8
        if sys.version_info.minor == 9:
            return stdlib3_9
        if sys.version_info.minor >= 10:
            return sys.stdlib_module_names
    elif sys.version_info.major > 3:
        return sys.stdlib_module_names
    raise RuntimeError(f'Unsupported Python version: {sys.version_info}')