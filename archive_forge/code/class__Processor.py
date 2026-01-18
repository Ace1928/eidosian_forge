import collections
import os
import re
import sys
import functools
import itertools
class _Processor:

    @classmethod
    def get(cls):
        func = getattr(cls, f'get_{sys.platform}', cls.from_subprocess)
        return func() or ''

    def get_win32():
        return os.environ.get('PROCESSOR_IDENTIFIER', _get_machine_win32())

    def get_OpenVMS():
        try:
            import vms_lib
        except ImportError:
            pass
        else:
            csid, cpu_number = vms_lib.getsyi('SYI$_CPU', 0)
            return 'Alpha' if cpu_number >= 128 else 'VAX'

    def from_subprocess():
        """
        Fall back to `uname -p`
        """
        try:
            import subprocess
        except ImportError:
            return None
        try:
            return subprocess.check_output(['uname', '-p'], stderr=subprocess.DEVNULL, text=True, encoding='utf8').strip()
        except (OSError, subprocess.CalledProcessError):
            pass