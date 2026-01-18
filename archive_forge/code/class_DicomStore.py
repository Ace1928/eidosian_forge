from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DicomStore(_messages.Message):
    """Represents a DICOM store.

  Messages:
    LabelsValue: User-supplied key-value pairs used to organize DICOM stores.
      Label keys must be between 1 and 63 characters long, have a UTF-8
      encoding of maximum 128 bytes, and must conform to the following PCRE
      regular expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must
      be between 1 and 63 characters long, have a UTF-8 encoding of maximum
      128 bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.

  Fields:
    labels: User-supplied key-value pairs used to organize DICOM stores. Label
      keys must be between 1 and 63 characters long, have a UTF-8 encoding of
      maximum 128 bytes, and must conform to the following PCRE regular
      expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must be
      between 1 and 63 characters long, have a UTF-8 encoding of maximum 128
      bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.
    name: Identifier. Resource name of the DICOM store, of the form `projects/
      {project_id}/locations/{location_id}/datasets/{dataset_id}/dicomStores/{
      dicom_store_id}`.
    notificationConfig: Notification destination for new DICOM instances.
      Supplied by the client.
    streamConfigs: Optional. A list of streaming configs used to configure the
      destination of streaming exports for every DICOM instance insertion in
      this DICOM store. After a new config is added to `stream_configs`, DICOM
      instance insertions are streamed to the new destination. When a config
      is removed from `stream_configs`, the server stops streaming to that
      destination. Each config must contain a unique destination.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-supplied key-value pairs used to organize DICOM stores. Label
    keys must be between 1 and 63 characters long, have a UTF-8 encoding of
    maximum 128 bytes, and must conform to the following PCRE regular
    expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must be between
    1 and 63 characters long, have a UTF-8 encoding of maximum 128 bytes, and
    must conform to the following PCRE regular expression:
    [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated with
    a given store.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    name = _messages.StringField(2)
    notificationConfig = _messages.MessageField('NotificationConfig', 3)
    streamConfigs = _messages.MessageField('GoogleCloudHealthcareV1alpha2DicomStreamConfig', 4, repeated=True)