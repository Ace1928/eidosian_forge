import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
class DarwinCPUInfo(CPUInfoBase):
    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = command_info(arch='arch', machine='machine')
        info['sysctl_hw'] = key_value_from_command('sysctl hw', sep='=')
        self.__class__.info = info

    def _not_impl(self):
        pass

    def _getNCPUs(self):
        return int(self.info['sysctl_hw'].get('hw.ncpu', 1))

    def _is_Power_Macintosh(self):
        return self.info['sysctl_hw']['hw.machine'] == 'Power Macintosh'

    def _is_i386(self):
        return self.info['arch'] == 'i386'

    def _is_ppc(self):
        return self.info['arch'] == 'ppc'

    def __machine(self, n):
        return self.info['machine'] == 'ppc%s' % n

    def _is_ppc601(self):
        return self.__machine(601)

    def _is_ppc602(self):
        return self.__machine(602)

    def _is_ppc603(self):
        return self.__machine(603)

    def _is_ppc603e(self):
        return self.__machine('603e')

    def _is_ppc604(self):
        return self.__machine(604)

    def _is_ppc604e(self):
        return self.__machine('604e')

    def _is_ppc620(self):
        return self.__machine(620)

    def _is_ppc630(self):
        return self.__machine(630)

    def _is_ppc740(self):
        return self.__machine(740)

    def _is_ppc7400(self):
        return self.__machine(7400)

    def _is_ppc7450(self):
        return self.__machine(7450)

    def _is_ppc750(self):
        return self.__machine(750)

    def _is_ppc403(self):
        return self.__machine(403)

    def _is_ppc505(self):
        return self.__machine(505)

    def _is_ppc801(self):
        return self.__machine(801)

    def _is_ppc821(self):
        return self.__machine(821)

    def _is_ppc823(self):
        return self.__machine(823)

    def _is_ppc860(self):
        return self.__machine(860)