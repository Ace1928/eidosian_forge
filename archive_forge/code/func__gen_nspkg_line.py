import os
from distutils import log
import itertools
def _gen_nspkg_line(self, pkg):
    pth = tuple(pkg.split('.'))
    root = self._get_root()
    tmpl_lines = self._nspkg_tmpl
    parent, sep, child = pkg.rpartition('.')
    if parent:
        tmpl_lines += self._nspkg_tmpl_multi
    return ';'.join(tmpl_lines) % locals() + '\n'