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
class Positional(FlagOrPositional):
    """Positional info."""

    def __init__(self, positional, name):
        super(Positional, self).__init__(positional, name)
        self.is_positional = True
        if positional.nargs is None:
            self.nargs = '1'
        self.is_required = positional.nargs not in (0, '?', '*', '...')