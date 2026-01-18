import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteSortedXcodeEnv(self, target, env):
    for k, v in env:
        self.WriteLn(f'{QuoteSpaces(target)}: export {k} := {v}')