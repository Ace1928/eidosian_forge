from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import MultiWidthBarcode
from string import digits
def _try_TO_C(self, l):
    """Improved version of old _trailingDigitsToC(self, l) inspired by"""
    i = 0
    nl = []
    while i < len(l):
        startpos = i
        rl = []
        savings = -1
        while i < len(l):
            if l[i] in cStarts:
                j = i
                break
            elif l[i] == 'ñ':
                rl.append(l[i])
                i += 1
                continue
            elif l[i] in digits and l[i + 1] in digits:
                rl.append(l[i] + l[i + 1])
                i += 2
                savings += 1
                continue
            else:
                if l[i] in digits and l[i + 1] == 'STOP':
                    rrl = []
                    rsavings = -1
                    k = i
                    while k > startpos:
                        if l[k] == 'ñ':
                            rrl.append(l[i])
                            k -= 1
                        elif l[k] in digits and l[k - 1] in digits:
                            rrl.append(l[k - 1] + l[k])
                            rsavings += 1
                            k -= 2
                        else:
                            break
                    rrl.reverse()
                    if rsavings > savings + int(savings >= 0 and (startpos and nl[-1] in cStarts)) - 1:
                        nl += l[startpos]
                        startpos += 1
                        rl = rrl
                        del rrl
                        i += 1
                break
        ta = not (l[i] == 'STOP' or j == i)
        xs = savings >= 0 and (startpos and nl[-1] in cStarts)
        if savings + int(xs) > int(ta):
            if xs:
                toc = nl[-1][:-1] + 'C'
                del nl[-1]
            else:
                toc = 'TO_C'
            nl += [toc] + rl
            if ta:
                nl.append('TO' + l[j][-2:])
            nl.append(l[i])
        else:
            nl += l[startpos:i + 1]
        i += 1
    return nl