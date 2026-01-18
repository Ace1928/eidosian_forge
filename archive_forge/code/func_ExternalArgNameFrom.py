from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
def ExternalArgNameFrom(arg_internal_name):
    """Converts an internal arg name into its corresponding user-visible name.

  This is used for creating exceptions using user-visible arg names.

  Args:
    arg_internal_name: the internal name of an argument.

  Returns:
    The user visible name for the argument.
  """
    if arg_internal_name == 'async_':
        return 'async'
    return arg_internal_name.replace('_', '-')