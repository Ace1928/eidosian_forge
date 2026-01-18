import os
def _check_cmd(cmd):
    safe_chars = []
    for first, last in (('a', 'z'), ('A', 'Z'), ('0', '9')):
        for ch in range(ord(first), ord(last) + 1):
            safe_chars.append(chr(ch))
    safe_chars.append('./-')
    safe_chars = ''.join(safe_chars)
    if isinstance(cmd, (tuple, list)):
        check_strs = cmd
    elif isinstance(cmd, str):
        check_strs = [cmd]
    else:
        return False
    for arg in check_strs:
        if not isinstance(arg, str):
            return False
        if not arg:
            return False
        for ch in arg:
            if ch not in safe_chars:
                return False
    return True