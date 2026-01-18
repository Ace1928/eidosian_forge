from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import flags
from googlecloudsdk.command_lib.iam.byoid_utilities import cred_config
Create a configuration file for generated credentials.

  This command creates a configuration file to allow access to authenticated
  Google Cloud actions from a variety of external user accounts.
  