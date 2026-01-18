import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
class LinuxCPUInfo(CPUInfoBase):
    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = [{}]
        ok, output = getoutput('uname -m')
        if ok:
            info[0]['uname_m'] = output.strip()
        try:
            fo = open('/proc/cpuinfo')
        except OSError as e:
            warnings.warn(str(e), UserWarning, stacklevel=2)
        else:
            for line in fo:
                name_value = [s.strip() for s in line.split(':', 1)]
                if len(name_value) != 2:
                    continue
                name, value = name_value
                if not info or name in info[-1]:
                    info.append({})
                info[-1][name] = value
            fo.close()
        self.__class__.info = info

    def _not_impl(self):
        pass

    def _is_AMD(self):
        return self.info[0]['vendor_id'] == 'AuthenticAMD'

    def _is_AthlonK6_2(self):
        return self._is_AMD() and self.info[0]['model'] == '2'

    def _is_AthlonK6_3(self):
        return self._is_AMD() and self.info[0]['model'] == '3'

    def _is_AthlonK6(self):
        return re.match('.*?AMD-K6', self.info[0]['model name']) is not None

    def _is_AthlonK7(self):
        return re.match('.*?AMD-K7', self.info[0]['model name']) is not None

    def _is_AthlonMP(self):
        return re.match('.*?Athlon\\(tm\\) MP\\b', self.info[0]['model name']) is not None

    def _is_AMD64(self):
        return self.is_AMD() and self.info[0]['family'] == '15'

    def _is_Athlon64(self):
        return re.match('.*?Athlon\\(tm\\) 64\\b', self.info[0]['model name']) is not None

    def _is_AthlonHX(self):
        return re.match('.*?Athlon HX\\b', self.info[0]['model name']) is not None

    def _is_Opteron(self):
        return re.match('.*?Opteron\\b', self.info[0]['model name']) is not None

    def _is_Hammer(self):
        return re.match('.*?Hammer\\b', self.info[0]['model name']) is not None

    def _is_Alpha(self):
        return self.info[0]['cpu'] == 'Alpha'

    def _is_EV4(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV4'

    def _is_EV5(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV5'

    def _is_EV56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV56'

    def _is_PCA56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'PCA56'
    _is_i386 = _not_impl

    def _is_Intel(self):
        return self.info[0]['vendor_id'] == 'GenuineIntel'

    def _is_i486(self):
        return self.info[0]['cpu'] == 'i486'

    def _is_i586(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '5'

    def _is_i686(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '6'

    def _is_Celeron(self):
        return re.match('.*?Celeron', self.info[0]['model name']) is not None

    def _is_Pentium(self):
        return re.match('.*?Pentium', self.info[0]['model name']) is not None

    def _is_PentiumII(self):
        return re.match('.*?Pentium.*?II\\b', self.info[0]['model name']) is not None

    def _is_PentiumPro(self):
        return re.match('.*?PentiumPro\\b', self.info[0]['model name']) is not None

    def _is_PentiumMMX(self):
        return re.match('.*?Pentium.*?MMX\\b', self.info[0]['model name']) is not None

    def _is_PentiumIII(self):
        return re.match('.*?Pentium.*?III\\b', self.info[0]['model name']) is not None

    def _is_PentiumIV(self):
        return re.match('.*?Pentium.*?(IV|4)\\b', self.info[0]['model name']) is not None

    def _is_PentiumM(self):
        return re.match('.*?Pentium.*?M\\b', self.info[0]['model name']) is not None

    def _is_Prescott(self):
        return self.is_PentiumIV() and self.has_sse3()

    def _is_Nocona(self):
        return self.is_Intel() and (self.info[0]['cpu family'] == '6' or self.info[0]['cpu family'] == '15') and (self.has_sse3() and (not self.has_ssse3())) and (re.match('.*?\\blm\\b', self.info[0]['flags']) is not None)

    def _is_Core2(self):
        return self.is_64bit() and self.is_Intel() and (re.match('.*?Core\\(TM\\)2\\b', self.info[0]['model name']) is not None)

    def _is_Itanium(self):
        return re.match('.*?Itanium\\b', self.info[0]['family']) is not None

    def _is_XEON(self):
        return re.match('.*?XEON\\b', self.info[0]['model name'], re.IGNORECASE) is not None
    _is_Xeon = _is_XEON

    def _is_singleCPU(self):
        return len(self.info) == 1

    def _getNCPUs(self):
        return len(self.info)

    def _has_fdiv_bug(self):
        return self.info[0]['fdiv_bug'] == 'yes'

    def _has_f00f_bug(self):
        return self.info[0]['f00f_bug'] == 'yes'

    def _has_mmx(self):
        return re.match('.*?\\bmmx\\b', self.info[0]['flags']) is not None

    def _has_sse(self):
        return re.match('.*?\\bsse\\b', self.info[0]['flags']) is not None

    def _has_sse2(self):
        return re.match('.*?\\bsse2\\b', self.info[0]['flags']) is not None

    def _has_sse3(self):
        return re.match('.*?\\bpni\\b', self.info[0]['flags']) is not None

    def _has_ssse3(self):
        return re.match('.*?\\bssse3\\b', self.info[0]['flags']) is not None

    def _has_3dnow(self):
        return re.match('.*?\\b3dnow\\b', self.info[0]['flags']) is not None

    def _has_3dnowext(self):
        return re.match('.*?\\b3dnowext\\b', self.info[0]['flags']) is not None