import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
class Win32CPUInfo(CPUInfoBase):
    info = None
    pkey = 'HARDWARE\\DESCRIPTION\\System\\CentralProcessor'

    def __init__(self):
        if self.info is not None:
            return
        info = []
        try:
            import winreg
            prgx = re.compile('family\\s+(?P<FML>\\d+)\\s+model\\s+(?P<MDL>\\d+)\\s+stepping\\s+(?P<STP>\\d+)', re.IGNORECASE)
            chnd = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.pkey)
            pnum = 0
            while True:
                try:
                    proc = winreg.EnumKey(chnd, pnum)
                except winreg.error:
                    break
                else:
                    pnum += 1
                    info.append({'Processor': proc})
                    phnd = winreg.OpenKey(chnd, proc)
                    pidx = 0
                    while True:
                        try:
                            name, value, vtpe = winreg.EnumValue(phnd, pidx)
                        except winreg.error:
                            break
                        else:
                            pidx = pidx + 1
                            info[-1][name] = value
                            if name == 'Identifier':
                                srch = prgx.search(value)
                                if srch:
                                    info[-1]['Family'] = int(srch.group('FML'))
                                    info[-1]['Model'] = int(srch.group('MDL'))
                                    info[-1]['Stepping'] = int(srch.group('STP'))
        except Exception as e:
            print(e, '(ignoring)')
        self.__class__.info = info

    def _not_impl(self):
        pass

    def _is_AMD(self):
        return self.info[0]['VendorIdentifier'] == 'AuthenticAMD'

    def _is_Am486(self):
        return self.is_AMD() and self.info[0]['Family'] == 4

    def _is_Am5x86(self):
        return self.is_AMD() and self.info[0]['Family'] == 4

    def _is_AMDK5(self):
        return self.is_AMD() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] in [0, 1, 2, 3])

    def _is_AMDK6(self):
        return self.is_AMD() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] in [6, 7])

    def _is_AMDK6_2(self):
        return self.is_AMD() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] == 8)

    def _is_AMDK6_3(self):
        return self.is_AMD() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] == 9)

    def _is_AMDK7(self):
        return self.is_AMD() and self.info[0]['Family'] == 6

    def _is_AMD64(self):
        return self.is_AMD() and self.info[0]['Family'] == 15

    def _is_Intel(self):
        return self.info[0]['VendorIdentifier'] == 'GenuineIntel'

    def _is_i386(self):
        return self.info[0]['Family'] == 3

    def _is_i486(self):
        return self.info[0]['Family'] == 4

    def _is_i586(self):
        return self.is_Intel() and self.info[0]['Family'] == 5

    def _is_i686(self):
        return self.is_Intel() and self.info[0]['Family'] == 6

    def _is_Pentium(self):
        return self.is_Intel() and self.info[0]['Family'] == 5

    def _is_PentiumMMX(self):
        return self.is_Intel() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] == 4)

    def _is_PentiumPro(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 and (self.info[0]['Model'] == 1)

    def _is_PentiumII(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 and (self.info[0]['Model'] in [3, 5, 6])

    def _is_PentiumIII(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 and (self.info[0]['Model'] in [7, 8, 9, 10, 11])

    def _is_PentiumIV(self):
        return self.is_Intel() and self.info[0]['Family'] == 15

    def _is_PentiumM(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 and (self.info[0]['Model'] in [9, 13, 14])

    def _is_Core2(self):
        return self.is_Intel() and self.info[0]['Family'] == 6 and (self.info[0]['Model'] in [15, 16, 17])

    def _is_singleCPU(self):
        return len(self.info) == 1

    def _getNCPUs(self):
        return len(self.info)

    def _has_mmx(self):
        if self.is_Intel():
            return self.info[0]['Family'] == 5 and self.info[0]['Model'] == 4 or self.info[0]['Family'] in [6, 15]
        elif self.is_AMD():
            return self.info[0]['Family'] in [5, 6, 15]
        else:
            return False

    def _has_sse(self):
        if self.is_Intel():
            return self.info[0]['Family'] == 6 and self.info[0]['Model'] in [7, 8, 9, 10, 11] or self.info[0]['Family'] == 15
        elif self.is_AMD():
            return self.info[0]['Family'] == 6 and self.info[0]['Model'] in [6, 7, 8, 10] or self.info[0]['Family'] == 15
        else:
            return False

    def _has_sse2(self):
        if self.is_Intel():
            return self.is_Pentium4() or self.is_PentiumM() or self.is_Core2()
        elif self.is_AMD():
            return self.is_AMD64()
        else:
            return False

    def _has_3dnow(self):
        return self.is_AMD() and self.info[0]['Family'] in [5, 6, 15]

    def _has_3dnowext(self):
        return self.is_AMD() and self.info[0]['Family'] in [6, 15]