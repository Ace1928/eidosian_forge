from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import enum
import getpass
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_pager
from googlecloudsdk.core.console import prompt_completer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def JsonUXStub(ux_type, **kwargs):
    """Generates a stub message for UX console output."""
    output = collections.OrderedDict()
    output['ux'] = ux_type.name
    extra_args = list(set(kwargs) - set(ux_type.GetDataFields()))
    if extra_args:
        raise ValueError('Extraneous args for Ux Element {}: {}'.format(ux_type.name, extra_args))
    for field in ux_type.GetDataFields():
        val = kwargs.get(field, None)
        if val:
            output[field] = val
    return json.dumps(output)