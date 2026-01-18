from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
def CopyV1Beta1(self, destination_region_ref=None, source_model=None, kms_key_name=None, destination_model_id=None, destination_parent_model=None):
    """Copies the given source model into specified location.

    The source model is copied into specified location (including cross-region)
    either as a new model or a new model version under given parent model.

    Args:
      destination_region_ref: the resource reference to the location into which
        to copy the Model.
      source_model: The resource name of the Model to copy.
      kms_key_name: The KMS key name for specifying encryption spec.
      destination_model_id: The destination model resource name to copy the
        model into.
      destination_parent_model: The destination parent model to copy the model
        as a model version into.

    Returns:
      Response from calling copy model.
    """
    encryption_spec = None
    if kms_key_name:
        encryption_spec = self.messages.GoogleCloudAiplatformV1beta1EncryptionSpec(kmsKeyName=kms_key_name)
    request = self.messages.AiplatformProjectsLocationsModelsCopyRequest(parent=destination_region_ref.RelativeName(), googleCloudAiplatformV1beta1CopyModelRequest=self.messages.GoogleCloudAiplatformV1beta1CopyModelRequest(sourceModel=source_model, encryptionSpec=encryption_spec, parentModel=destination_parent_model, modelId=destination_model_id))
    return self._service.Copy(request)