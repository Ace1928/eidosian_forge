import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteDoCmd(self, outputs, inputs, command, part_of_all, comment=None, postbuilds=False):
    """Write a Makefile rule that uses do_cmd.

        This makes the outputs dependent on the command line that was run,
        as well as support the V= make command line flag.
        """
    suffix = ''
    if postbuilds:
        assert ',' not in command
        suffix = ',,1'
    self.WriteMakeRule(outputs, inputs, actions=[f'$(call do_cmd,{command}{suffix})'], comment=comment, command=command, force=True)
    outputs = [QuoteSpaces(o, SPACE_REPLACEMENT) for o in outputs]
    self.WriteLn('all_deps += %s' % ' '.join(outputs))