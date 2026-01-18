import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _cc_normalize_unix(self, flags):

    def ver_flags(f):
        tokens = f.split('+')
        ver = float('0' + ''.join(re.findall(self._cc_normalize_arch_ver, tokens[0])))
        return (ver, tokens[0], tokens[1:])
    if len(flags) <= 1:
        return flags
    for i, cur_flag in enumerate(reversed(flags)):
        if not re.match(self._cc_normalize_unix_mrgx, cur_flag):
            continue
        lower_flags = flags[:-(i + 1)]
        upper_flags = flags[-i:]
        filtered = list(filter(self._cc_normalize_unix_frgx.search, lower_flags))
        ver, arch, subflags = ver_flags(cur_flag)
        if ver > 0 and len(subflags) > 0:
            for xflag in lower_flags:
                xver, _, xsubflags = ver_flags(xflag)
                if ver == xver:
                    subflags = xsubflags + subflags
            cur_flag = arch + '+' + '+'.join(subflags)
        flags = filtered + [cur_flag]
        if i > 0:
            flags += upper_flags
        break
    final_flags = []
    matched = set()
    for f in reversed(flags):
        match = re.match(self._cc_normalize_unix_krgx, f)
        if not match:
            pass
        elif match[0] in matched:
            continue
        else:
            matched.add(match[0])
        final_flags.insert(0, f)
    return final_flags