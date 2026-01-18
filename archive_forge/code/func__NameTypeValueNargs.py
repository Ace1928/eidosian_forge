from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
def _NameTypeValueNargs(name, type_=None, value=None, nargs=None):
    """Returns the (name, type, value-metavar, nargs) for flag name."""
    if name.startswith('--'):
        if '=' in name:
            name, value = name.split('=', 1)
            if name[-1] == '[':
                name = name[:-1]
                if value.endswith(']'):
                    value = value[:-1]
                nargs = '?'
            else:
                nargs = '1'
            type_ = 'string'
    elif len(name) > 2:
        value = name[2:]
        if value[0].isspace():
            value = value[1:]
        if value.startswith('['):
            value = value[1:]
            if value.endswith(']'):
                value = value[:-1]
            nargs = '?'
        else:
            nargs = '1'
        name = name[:2]
        type_ = 'string'
    if type_ is None or value is None or nargs is None:
        type_ = 'bool'
        value = ''
        nargs = '0'
    return (name, type_, value, nargs)