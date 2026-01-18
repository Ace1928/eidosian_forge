import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def ComputeOutput(self, spec):
    """Return the 'output' (full output path) of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          '$(obj)/baz/libfoobar.so'
        """
    assert not self.is_mac_bundle
    path = os.path.join('$(obj).' + self.toolset, self.path)
    if self.type == 'executable' or self._InstallImmediately():
        path = '$(builddir)'
    path = spec.get('product_dir', path)
    return os.path.join(path, self.ComputeOutputBasename(spec))