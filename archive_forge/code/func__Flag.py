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
def _Flag(name, description='', value=None, default=None, type_='string', category='', is_global=False, is_required=False, nargs=None):
    """Initializes and returns a flag dict node."""
    return {cli_tree.LOOKUP_ATTR: {}, cli_tree.LOOKUP_CATEGORY: category, cli_tree.LOOKUP_DEFAULT: default, cli_tree.LOOKUP_DESCRIPTION: _NormalizeSpace(description), cli_tree.LOOKUP_GROUP: '', cli_tree.LOOKUP_IS_GLOBAL: is_global, cli_tree.LOOKUP_IS_HIDDEN: False, cli_tree.LOOKUP_IS_REQUIRED: is_required, cli_tree.LOOKUP_NAME: name, cli_tree.LOOKUP_NARGS: nargs or ('0' if type_ == 'bool' else '1'), cli_tree.LOOKUP_VALUE: value, cli_tree.LOOKUP_TYPE: type_}