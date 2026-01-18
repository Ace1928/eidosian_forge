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
def _Add(name, description, category, type_, value, nargs):
    """Adds a flag."""
    name, type_, value, nargs = _NameTypeValueNargs(name, type_, value, nargs)
    default = ''
    command[cli_tree.LOOKUP_FLAGS][name] = _Flag(name=name, description='\n'.join(description), type_=type_, value=value, nargs=nargs, category=category, default=default, is_required=False, is_global=is_global)