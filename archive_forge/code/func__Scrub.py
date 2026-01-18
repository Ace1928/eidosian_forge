from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _Scrub(self):
    """Scrubs private paths in the default value and description.

    Argument default values and "The default is ..." description text are the
    only places where dynamic private file paths can leak into the cli_tree.
    This method is called on all args.

    The test is rudimentary but effective. Any default value that looks like an
    absolute path on unix or windows is scrubbed. The default value is set to
    None and the trailing "The default ... is ..." sentence in the description,
    if any, is deleted. It's OK to be conservative here and match aggressively.
    """
    if not isinstance(self.default, six.string_types):
        return
    if not re.match('/|[A-Za-z]:\\\\', self.default):
        return
    self.default = None
    match = re.match('(.*\\.) The default (value )?is ', self.description, re.DOTALL)
    if match:
        self.description = match.group(1)