from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam.byoid_utilities import cred_config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
Create a login configuration file to enable sign-in via a web-based authorization flow using Workforce Identity Federation.

  This command creates a configuration file to enable browser based
  third-party sign in with Workforce Identity Federation through
  `gcloud auth login --login-config=/path/to/config.json`.
  