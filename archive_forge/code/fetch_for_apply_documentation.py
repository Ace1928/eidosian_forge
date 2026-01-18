from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import semver
Merge configSync set in feature spec with the config template.

  ConfigSync has nested object structs need to be flatten.

  Args:
    spec: the ConfigManagementMembershipSpec message
    config: the dict loaded from full config template
    version: the version string of the membership
  