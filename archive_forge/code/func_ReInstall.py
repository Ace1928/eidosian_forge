from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import bootstrapping
import argparse
import os
import sys
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import platforms_install
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk import gcloud_main
def ReInstall(component_ids, compile_python):
    """Do a forced reinstallation of Google Cloud CLI.

  Args:
    component_ids: [str], The components that should be automatically installed.
    compile_python: bool, False if we skip compile python
  """
    to_install = bootstrapping.GetDefaultInstalledComponents()
    to_install.extend(component_ids)
    InstallOrUpdateComponents(component_ids, compile_python, update=True)