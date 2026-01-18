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
def _FlagIndexKey(flag):
    return '::'.join([six.text_type(flag.name), six.text_type(flag.attr), six.text_type(flag.category), '[{}]'.format(', '.join((six.text_type(c) for c in flag.choices))), six.text_type(flag.completer), six.text_type(flag.default), six.text_type(flag.description), six.text_type(flag.is_hidden), six.text_type(flag.is_global), six.text_type(flag.is_group), six.text_type(flag.is_required), six.text_type(flag.nargs), six.text_type(flag.type), six.text_type(flag.value)])