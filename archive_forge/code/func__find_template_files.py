import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def _find_template_files(self, template, tmpl_dir, vars, file_sources, join=''):
    full_dir = os.path.join(tmpl_dir, join)
    for name in os.listdir(full_dir):
        if name.startswith('.'):
            continue
        if os.path.isdir(os.path.join(full_dir, name)):
            self._find_template_files(template, tmpl_dir, vars, file_sources, join=os.path.join(join, name))
            continue
        partial = os.path.join(join, name)
        for name, value in vars.items():
            partial = partial.replace('+%s+' % name, value)
        if partial.endswith('_tmpl'):
            partial = partial[:-5]
        file_sources.setdefault(partial, []).append(template)