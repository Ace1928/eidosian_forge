import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def _show_files(self, output_dir, file_sources, join='', indent=0):
    pad = ' ' * (2 * indent)
    full_dir = os.path.join(output_dir, join)
    names = os.listdir(full_dir)
    dirs = [n for n in names if os.path.isdir(os.path.join(full_dir, n))]
    fns = [n for n in names if not os.path.isdir(os.path.join(full_dir, n))]
    dirs.sort()
    names.sort()
    for name in names:
        skip_this = False
        for ext in self._ignore_filenames:
            if fnmatch.fnmatch(name, ext):
                if self.verbose > 1:
                    print('%sIgnoring %s' % (pad, name))
                skip_this = True
                break
        if skip_this:
            continue
        partial = os.path.join(join, name)
        if partial not in file_sources:
            if self.verbose > 1:
                print('%s%s (not from template)' % (pad, name))
            continue
        templates = file_sources.pop(partial)
        print('%s%s from:' % (pad, name))
        for template in templates:
            print('%s  %s' % (pad, template.name))
    for dir in dirs:
        if dir in self._ignore_dirs:
            continue
        print('%sRecursing into %s/' % (pad, dir))
        self._show_files(output_dir, file_sources, join=os.path.join(join, dir), indent=indent + 1)