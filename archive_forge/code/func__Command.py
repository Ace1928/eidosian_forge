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
def _Command(path):
    """Initializes and returns a command/group dict node."""
    return {cli_tree.LOOKUP_CAPSULE: '', cli_tree.LOOKUP_COMMANDS: {}, cli_tree.LOOKUP_FLAGS: {}, cli_tree.LOOKUP_GROUPS: {}, cli_tree.LOOKUP_IS_GROUP: False, cli_tree.LOOKUP_IS_HIDDEN: False, cli_tree.LOOKUP_PATH: path, cli_tree.LOOKUP_POSITIONALS: [], cli_tree.LOOKUP_RELEASE: 'GA', cli_tree.LOOKUP_SECTIONS: {}}