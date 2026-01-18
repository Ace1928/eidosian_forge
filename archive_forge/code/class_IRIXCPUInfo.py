import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
class IRIXCPUInfo(CPUInfoBase):
    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = key_value_from_command('sysconf', sep=' ', successful_status=(0, 1))
        self.__class__.info = info

    def _not_impl(self):
        pass

    def _is_singleCPU(self):
        return self.info.get('NUM_PROCESSORS') == '1'

    def _getNCPUs(self):
        return int(self.info.get('NUM_PROCESSORS', 1))

    def __cputype(self, n):
        return self.info.get('PROCESSORS').split()[0].lower() == 'r%s' % n

    def _is_r2000(self):
        return self.__cputype(2000)

    def _is_r3000(self):
        return self.__cputype(3000)

    def _is_r3900(self):
        return self.__cputype(3900)

    def _is_r4000(self):
        return self.__cputype(4000)

    def _is_r4100(self):
        return self.__cputype(4100)

    def _is_r4300(self):
        return self.__cputype(4300)

    def _is_r4400(self):
        return self.__cputype(4400)

    def _is_r4600(self):
        return self.__cputype(4600)

    def _is_r4650(self):
        return self.__cputype(4650)

    def _is_r5000(self):
        return self.__cputype(5000)

    def _is_r6000(self):
        return self.__cputype(6000)

    def _is_r8000(self):
        return self.__cputype(8000)

    def _is_r10000(self):
        return self.__cputype(10000)

    def _is_r12000(self):
        return self.__cputype(12000)

    def _is_rorion(self):
        return self.__cputype('orion')

    def get_ip(self):
        try:
            return self.info.get('MACHINE')
        except Exception:
            pass

    def __machine(self, n):
        return self.info.get('MACHINE').lower() == 'ip%s' % n

    def _is_IP19(self):
        return self.__machine(19)

    def _is_IP20(self):
        return self.__machine(20)

    def _is_IP21(self):
        return self.__machine(21)

    def _is_IP22(self):
        return self.__machine(22)

    def _is_IP22_4k(self):
        return self.__machine(22) and self._is_r4000()

    def _is_IP22_5k(self):
        return self.__machine(22) and self._is_r5000()

    def _is_IP24(self):
        return self.__machine(24)

    def _is_IP25(self):
        return self.__machine(25)

    def _is_IP26(self):
        return self.__machine(26)

    def _is_IP27(self):
        return self.__machine(27)

    def _is_IP28(self):
        return self.__machine(28)

    def _is_IP30(self):
        return self.__machine(30)

    def _is_IP32(self):
        return self.__machine(32)

    def _is_IP32_5k(self):
        return self.__machine(32) and self._is_r5000()

    def _is_IP32_10k(self):
        return self.__machine(32) and self._is_r10000()