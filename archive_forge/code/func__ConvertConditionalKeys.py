import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _ConvertConditionalKeys(self, configname):
    """Converts or warns on conditional keys.  Xcode supports conditional keys,
    such as CODE_SIGN_IDENTITY[sdk=iphoneos*].  This is a partial implementation
    with some keys converted while the rest force a warning."""
    settings = self.xcode_settings[configname]
    conditional_keys = [key for key in settings if key.endswith(']')]
    for key in conditional_keys:
        if key.endswith('[sdk=iphoneos*]'):
            if configname.endswith('iphoneos'):
                new_key = key.split('[')[0]
                settings[new_key] = settings[key]
        else:
            print('Warning: Conditional keys not implemented, ignoring:', ' '.join(conditional_keys))
        del settings[key]