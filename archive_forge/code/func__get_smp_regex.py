import re
import sys
from builtins import str, chr
def _get_smp_regex():
    xls = sorted((x - 65536 for x in XLAT if x >= 65536))
    xls.append(-1)
    fmt, (dsh, opn, pip, cse) = (str('\\u%04x'), str('-[|]'))
    rga, srk, erk = ([str('[ \\t\\r\\n]+')], 0, -2)
    for k in xls:
        new_hir = (erk ^ k) >> 10 != 0
        if new_hir or erk + 1 != k:
            if erk >= 0 and srk != erk:
                if srk + 1 != erk:
                    rga.append(dsh)
                rga.append(fmt % (56320 + (erk & 1023)))
            if new_hir:
                if erk >= 0:
                    rga.append(cse)
                if k < 0:
                    break
                rga.append(pip)
                rga.append(fmt % (55296 + (k >> 10)))
                rga.append(opn)
            srk = k
            rga.append(fmt % (56320 + (srk & 1023)))
        erk = k
    return re.compile(str().join(rga))