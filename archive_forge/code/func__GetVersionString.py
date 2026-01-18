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
def _GetVersionString(component, installed_components):
    """Return most accurate VersionString for component with id comp_id.

  Component versions become stale in the case where only the architecture
  specific component has content. In these cases show the version from the
  architecture specific component, not the parent component.

  Args:
    component: updater.local_state.InstallationManifest of component to get
      VersionString of.
    installed_components: map of str to InstallationManifest of installed
      components.

  Returns:
    str, Most accurate VersionString for comp_id.
  """
    comp_def = component.ComponentDefinition()
    if comp_def.data is not None:
        return component.VersionString()
    try:
        for dep in [installed_components[d] for d in comp_def.dependencies if d in installed_components]:
            dep_def = dep.ComponentDefinition()
            if component.id in dep_def.dependencies and dep_def.platform.architectures:
                return dep.VersionString()
    except:
        pass
    return component.VersionString()