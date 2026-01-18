import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
def ComputeOutputParts(self, spec):
    """Return the 'output basename' of a gyp spec, split into filename + ext.

        Android libraries must be named the same thing as their module name,
        otherwise the linker can't find them, so product_name and so on must be
        ignored if we are building a library, and the "lib" prepending is
        not done for Android.
        """
    assert self.type != 'loadable_module'
    target = spec['target_name']
    target_prefix = ''
    target_ext = ''
    if self.type == 'static_library':
        target = self.ComputeAndroidModule(spec)
        target_ext = '.a'
    elif self.type == 'shared_library':
        target = self.ComputeAndroidModule(spec)
        target_ext = '.so'
    elif self.type == 'none':
        target_ext = '.stamp'
    elif self.type != 'executable':
        print('ERROR: What output file should be generated?', 'type', self.type, 'target', target)
    if self.type != 'static_library' and self.type != 'shared_library':
        target_prefix = spec.get('product_prefix', target_prefix)
        target = spec.get('product_name', target)
        product_ext = spec.get('product_extension')
        if product_ext:
            target_ext = '.' + product_ext
    target_stem = target_prefix + target
    return (target_stem, target_ext)