from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.calliope.exceptions import core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
from mako import runtime
from mako import template
def ConstructModuleParameters(import_path, dest_dir):
    module_source = os.path.join('.', os.path.relpath(import_path, start=dest_dir))
    module_name = '-'.join(os.path.normpath(import_path.replace(dest_dir, '')).split(os.sep)).lstrip('-').rstrip()
    if module_name[0].isdigit():
        module_name = 'gcp-{}'.format(module_name)
    return (module_source, module_name)