from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class FilesValue(_messages.Message):
    """Manifest of the files stored in Google Cloud Storage that are included
    as part of this version. All files must be readable using the credentials
    supplied with this call.

    Messages:
      AdditionalProperty: An additional property for a FilesValue object.

    Fields:
      additionalProperties: Additional properties of type FilesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a FilesValue object.

      Fields:
        key: Name of the additional property.
        value: A FileInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('FileInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)