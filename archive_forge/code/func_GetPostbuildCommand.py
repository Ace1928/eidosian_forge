import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def GetPostbuildCommand(self, spec, output, output_binary, is_command_start):
    """Returns a shell command that runs all the postbuilds, and removes
        |output| if any of them fails. If |is_command_start| is False, then the
        returned string will start with ' && '."""
    if not self.xcode_settings or spec['type'] == 'none' or (not output):
        return ''
    output = QuoteShellArgument(output, self.flavor)
    postbuilds = gyp.xcode_emulation.GetSpecPostbuildCommands(spec, quiet=True)
    if output_binary is not None:
        postbuilds = self.xcode_settings.AddImplicitPostbuilds(self.config_name, os.path.normpath(os.path.join(self.base_to_build, output)), QuoteShellArgument(os.path.normpath(os.path.join(self.base_to_build, output_binary)), self.flavor), postbuilds, quiet=True)
    if not postbuilds:
        return ''
    postbuilds.insert(0, gyp.common.EncodePOSIXShellList(['cd', self.build_to_base]))
    env = self.ComputeExportEnvString(self.GetSortedXcodePostbuildEnv())
    commands = env + ' (' + ' && '.join([ninja_syntax.escape(command) for command in postbuilds])
    command_string = commands + '); G=$$?; ((exit $$G) || rm -rf %s) ' % output + '&& exit $$G)'
    if is_command_start:
        return '(' + command_string + ' && '
    else:
        return '$ && (' + command_string