from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.util.args import labels_util
def CreateGa(self, location_ref, args):
    """Create a new Tensorboard."""
    kms_key_name = common_validation.GetAndValidateKmsKey(args)
    labels = labels_util.ParseCreateArgs(args, self.messages.GoogleCloudAiplatformV1Tensorboard.LabelsValue)
    tensorboard = self.messages.GoogleCloudAiplatformV1Tensorboard(displayName=args.display_name, description=args.description, labels=labels)
    if kms_key_name is not None:
        tensorboard.encryptionSpec = api_util.GetMessage('EncryptionSpec', self._version)(kmsKeyName=kms_key_name)
    request = self.messages.AiplatformProjectsLocationsTensorboardsCreateRequest(parent=location_ref.RelativeName(), googleCloudAiplatformV1Tensorboard=tensorboard)
    return self._service.Create(request)