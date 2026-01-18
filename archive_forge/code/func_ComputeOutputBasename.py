import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def ComputeOutputBasename(self, spec):
    """Return the 'output basename' of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          'libfoobar.so'
        """
    assert not self.is_mac_bundle
    if self.flavor == 'mac' and self.type in ('static_library', 'executable', 'shared_library', 'loadable_module'):
        return self.xcode_settings.GetExecutablePath()
    target = spec['target_name']
    target_prefix = ''
    target_ext = ''
    if self.type == 'static_library':
        if target[:3] == 'lib':
            target = target[3:]
        target_prefix = 'lib'
        target_ext = '.a'
    elif self.type in ('loadable_module', 'shared_library'):
        if target[:3] == 'lib':
            target = target[3:]
        target_prefix = 'lib'
        if self.flavor == 'aix':
            target_ext = '.a'
        else:
            target_ext = '.so'
    elif self.type == 'none':
        target = '%s.stamp' % target
    elif self.type != 'executable':
        print('ERROR: What output file should be generated?', 'type', self.type, 'target', target)
    target_prefix = spec.get('product_prefix', target_prefix)
    target = spec.get('product_name', target)
    product_ext = spec.get('product_extension')
    if product_ext:
        target_ext = '.' + product_ext
    return target_prefix + target + target_ext