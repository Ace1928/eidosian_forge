import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetWrapperExtension(self):
    """Returns the bundle extension (.app, .framework, .plugin, etc).  Only
    valid for bundles."""
    assert self._IsBundle()
    if self.spec['type'] in ('loadable_module', 'shared_library'):
        default_wrapper_extension = {'loadable_module': 'bundle', 'shared_library': 'framework'}[self.spec['type']]
        wrapper_extension = self.GetPerTargetSetting('WRAPPER_EXTENSION', default=default_wrapper_extension)
        return '.' + self.spec.get('product_extension', wrapper_extension)
    elif self.spec['type'] == 'executable':
        if self._IsIosAppExtension() or self._IsIosWatchKitExtension():
            return '.' + self.spec.get('product_extension', 'appex')
        else:
            return '.' + self.spec.get('product_extension', 'app')
    else:
        assert False, "Don't know extension for '{}', target '{}'".format(self.spec['type'], self.spec['target_name'])