from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationDetails(_messages.Message):
    """Details about a single transformation. This object contains a
  description of the transformation, information about whether the
  transformation was successfully applied, and the precise location where the
  transformation occurred. These details are stored in a user-specified
  BigQuery table.

  Fields:
    containerName: The top level name of the container where the
      transformation is located (this will be the source file name or table
      name).
    resourceName: The name of the job that completed the transformation.
    statusDetails: Status of the transformation, if transformation was not
      successful, this will specify what caused it to fail, otherwise it will
      show that the transformation was successful.
    transformation: Description of transformation. This would only contain
      more than one element if there were multiple matching transformations
      and which one to apply was ambiguous. Not set for states that contain no
      transformation, currently only state that contains no transformation is
      TransformationResultStateType.METADATA_UNRETRIEVABLE.
    transformationLocation: The precise location of the transformed content in
      the original container.
    transformedBytes: The number of bytes that were transformed. If
      transformation was unsuccessful or did not take place because there was
      no content to transform, this will be zero.
  """
    containerName = _messages.StringField(1)
    resourceName = _messages.StringField(2)
    statusDetails = _messages.MessageField('GooglePrivacyDlpV2TransformationResultStatus', 3)
    transformation = _messages.MessageField('GooglePrivacyDlpV2TransformationDescription', 4, repeated=True)
    transformationLocation = _messages.MessageField('GooglePrivacyDlpV2TransformationLocation', 5)
    transformedBytes = _messages.IntegerField(6)