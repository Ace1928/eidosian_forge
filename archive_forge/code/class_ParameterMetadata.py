from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParameterMetadata(_messages.Message):
    """Metadata for a specific parameter.

  Enums:
    ParamTypeValueValuesEnum: Optional. The type of the parameter. Used for
      selecting input picker.

  Messages:
    CustomMetadataValue: Optional. Additional metadata for describing this
      parameter.

  Fields:
    customMetadata: Optional. Additional metadata for describing this
      parameter.
    defaultValue: Optional. The default values will pre-populate the parameter
      with the given value from the proto. If default_value is left empty, the
      parameter will be populated with a default of the relevant type, e.g.
      false for a boolean.
    enumOptions: Optional. The options shown when ENUM ParameterType is
      specified.
    groupName: Optional. Specifies a group name for this parameter to be
      rendered under. Group header text will be rendered exactly as specified
      in this field. Only considered when parent_name is NOT provided.
    helpText: Required. The help text to display for the parameter.
    hiddenUi: Optional. Whether the parameter should be hidden in the UI.
    isOptional: Optional. Whether the parameter is optional. Defaults to
      false.
    label: Required. The label to display for the parameter.
    name: Required. The name of the parameter.
    paramType: Optional. The type of the parameter. Used for selecting input
      picker.
    parentName: Optional. Specifies the name of the parent parameter. Used in
      conjunction with 'parent_trigger_values' to make this parameter
      conditional (will only be rendered conditionally). Should be mappable to
      a ParameterMetadata.name field.
    parentTriggerValues: Optional. The value(s) of the 'parent_name' parameter
      which will trigger this parameter to be shown. If left empty, ANY non-
      empty value in parent_name will trigger this parameter to be shown. Only
      considered when this parameter is conditional (when 'parent_name' has
      been provided).
    regexes: Optional. Regexes that the parameter must match.
  """

    class ParamTypeValueValuesEnum(_messages.Enum):
        """Optional. The type of the parameter. Used for selecting input picker.

    Values:
      DEFAULT: Default input type.
      TEXT: The parameter specifies generic text input.
      GCS_READ_BUCKET: The parameter specifies a Cloud Storage Bucket to read
        from.
      GCS_WRITE_BUCKET: The parameter specifies a Cloud Storage Bucket to
        write to.
      GCS_READ_FILE: The parameter specifies a Cloud Storage file path to read
        from.
      GCS_WRITE_FILE: The parameter specifies a Cloud Storage file path to
        write to.
      GCS_READ_FOLDER: The parameter specifies a Cloud Storage folder path to
        read from.
      GCS_WRITE_FOLDER: The parameter specifies a Cloud Storage folder to
        write to.
      PUBSUB_TOPIC: The parameter specifies a Pub/Sub Topic.
      PUBSUB_SUBSCRIPTION: The parameter specifies a Pub/Sub Subscription.
      BIGQUERY_TABLE: The parameter specifies a BigQuery table.
      JAVASCRIPT_UDF_FILE: The parameter specifies a JavaScript UDF in Cloud
        Storage.
      SERVICE_ACCOUNT: The parameter specifies a Service Account email.
      MACHINE_TYPE: The parameter specifies a Machine Type.
      KMS_KEY_NAME: The parameter specifies a KMS Key name.
      WORKER_REGION: The parameter specifies a Worker Region.
      WORKER_ZONE: The parameter specifies a Worker Zone.
      BOOLEAN: The parameter specifies a boolean input.
      ENUM: The parameter specifies an enum input.
      NUMBER: The parameter specifies a number input.
    """
        DEFAULT = 0
        TEXT = 1
        GCS_READ_BUCKET = 2
        GCS_WRITE_BUCKET = 3
        GCS_READ_FILE = 4
        GCS_WRITE_FILE = 5
        GCS_READ_FOLDER = 6
        GCS_WRITE_FOLDER = 7
        PUBSUB_TOPIC = 8
        PUBSUB_SUBSCRIPTION = 9
        BIGQUERY_TABLE = 10
        JAVASCRIPT_UDF_FILE = 11
        SERVICE_ACCOUNT = 12
        MACHINE_TYPE = 13
        KMS_KEY_NAME = 14
        WORKER_REGION = 15
        WORKER_ZONE = 16
        BOOLEAN = 17
        ENUM = 18
        NUMBER = 19

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomMetadataValue(_messages.Message):
        """Optional. Additional metadata for describing this parameter.

    Messages:
      AdditionalProperty: An additional property for a CustomMetadataValue
        object.

    Fields:
      additionalProperties: Additional properties of type CustomMetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomMetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    customMetadata = _messages.MessageField('CustomMetadataValue', 1)
    defaultValue = _messages.StringField(2)
    enumOptions = _messages.MessageField('ParameterMetadataEnumOption', 3, repeated=True)
    groupName = _messages.StringField(4)
    helpText = _messages.StringField(5)
    hiddenUi = _messages.BooleanField(6)
    isOptional = _messages.BooleanField(7)
    label = _messages.StringField(8)
    name = _messages.StringField(9)
    paramType = _messages.EnumField('ParamTypeValueValuesEnum', 10)
    parentName = _messages.StringField(11)
    parentTriggerValues = _messages.StringField(12, repeated=True)
    regexes = _messages.StringField(13, repeated=True)