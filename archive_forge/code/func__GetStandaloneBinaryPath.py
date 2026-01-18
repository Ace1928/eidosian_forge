import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetStandaloneBinaryPath(self):
    """Returns the name of the non-bundle binary represented by this target.
    E.g. hello_world. Only valid for non-bundles."""
    assert not self._IsBundle()
    assert self.spec['type'] in ('executable', 'shared_library', 'static_library', 'loadable_module'), 'Unexpected type %s' % self.spec['type']
    target = self.spec['target_name']
    if self.spec['type'] == 'static_library':
        if target[:3] == 'lib':
            target = target[3:]
    elif self.spec['type'] in ('loadable_module', 'shared_library'):
        if target[:3] == 'lib':
            target = target[3:]
    target_prefix = self._GetStandaloneExecutablePrefix()
    target = self.spec.get('product_name', target)
    target_ext = self._GetStandaloneExecutableSuffix()
    return target_prefix + target + target_ext