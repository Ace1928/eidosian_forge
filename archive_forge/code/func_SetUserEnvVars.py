from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def SetUserEnvVars(env_vars, workflow, updated_fields):
    """Sets user-defined environment variables.

  Also updates updated_fields accordingly.

  Args:
    env_vars: Parsed environment variables to be set on the workflow.
    workflow: The workflow in which to set the User Envrionment Variables.
    updated_fields: A list to which the userEnvVars field will be added if
      needed.
  """
    if env_vars is None:
        return
    workflow.userEnvVars = None if env_vars is CLEAR_ENVIRONMENT else env_vars
    updated_fields.append('userEnvVars')