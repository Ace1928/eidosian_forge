from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def ValidateUpdateFieldMask(ref, unused_args, request):
    """Validate the field mask for an update request."""
    del ref, unused_args
    if not request.patchServiceAccountRequest.updateMask:
        update_fields = ['--display-name', '--description']
        raise gcloud_exceptions.OneOfArgumentsRequiredException(update_fields, 'Specify at least one field to update.')
    return request