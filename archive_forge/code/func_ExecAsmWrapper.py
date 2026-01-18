import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecAsmWrapper(self, arch, *args):
    """Filter logo banner from invocations of asm.exe."""
    env = self._GetEnv(arch)
    popen = subprocess.Popen(args, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = popen.communicate()[0].decode('utf-8')
    for line in out.splitlines():
        if not line.startswith('Copyright (C) Microsoft Corporation') and (not line.startswith('Microsoft (R) Macro Assembler')) and (not line.startswith(' Assembling: ')) and line:
            print(line)
    return popen.returncode