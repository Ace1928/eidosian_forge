import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def _find_files(self, template, vars, file_sources):
    tmpl_dir = template.template_dir()
    self._find_template_files(template, tmpl_dir, vars, file_sources)