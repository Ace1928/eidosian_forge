import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetPerTargetSettings(self):
    """Gets a list of all the per-target settings. This will only fetch keys
    whose values are the same across all configurations."""
    first_pass = True
    result = {}
    for configname in sorted(self.xcode_settings.keys()):
        if first_pass:
            result = dict(self.xcode_settings[configname])
            first_pass = False
        else:
            for key, value in self.xcode_settings[configname].items():
                if key not in result:
                    continue
                elif result[key] != value:
                    del result[key]
    return result