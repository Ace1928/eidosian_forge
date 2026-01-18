from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.kms import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def HandleErrors(self, args, set_primary_version_succeeds, other_updates_succeed, fields_to_update):
    """Handles various errors that may occur during any update stage.

    Never returns without an exception.

    Args:
      args: Input arguments.
      set_primary_version_succeeds: True if the primary verion is updated
        successfully.
      other_updates_succeed: True if all other updates (besides primary verions)
        is updated successfully.
      fields_to_update: A list of fields to be updated.

    Raises:
      ToolException: An exception raised when there is error during any update
      stage.
    """
    err = 'An Error occurred:'
    if not set_primary_version_succeeds:
        err += " Failed to update field 'primaryVersion'."
    elif args.primary_version:
        err += " Field 'primaryVersion' was updated."
    if not other_updates_succeed:
        err += " Failed to update field(s) '{}'.".format("', '".join(fields_to_update))
    elif fields_to_update:
        err += " Field(s) '{}' were updated.".format("', '".join(fields_to_update))
    raise kms_exceptions.UpdateError(err)