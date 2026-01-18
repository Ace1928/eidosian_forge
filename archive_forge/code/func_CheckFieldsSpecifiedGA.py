from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def CheckFieldsSpecifiedGA(unused_instance_ref, args, patch_request):
    """Checks if fields to update are registered for GA track."""
    additional_update_args = ['maintenance_version']
    return CheckFieldsSpecifiedCommon(args, patch_request, additional_update_args)