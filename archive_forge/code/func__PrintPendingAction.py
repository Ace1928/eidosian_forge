from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _PrintPendingAction(self, components, action):
    """Prints info about components we are going to install or remove.

    Args:
      components: list(schemas.Component), The components that are going to be
        acted on.
      action: str, The verb to print for this set of components.
    """
    attributes = ['box', 'title="These components will be {action}."'.format(action=action)]
    columns = ['details.display_name:label=Name:align=left', 'version.version_string:label=Version:align=right', 'data.size.size(zero="",min=1048576):label=Size:align=right']
    fmt = 'table[{attributes}]({columns})'.format(attributes=','.join(attributes), columns=','.join(columns))
    resource_printer.Print(components, fmt, out=log.status)