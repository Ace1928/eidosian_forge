from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes as compute_base
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Call Compute.Projects.Get to find project_id's default Service Account.