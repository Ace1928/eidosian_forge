from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SessionTemplate(_messages.Message):
    """A representation of a session template.

  Messages:
    LabelsValue: Optional. Labels to associate with sessions created using
      this template. Label keys must contain 1 to 63 characters, and must
      conform to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt). Label values
      can be empty, but, if present, must contain 1 to 63 characters and
      conform to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt). No more than
      32 labels can be associated with a session.

  Fields:
    createTime: Output only. The time when the template was created.
    creator: Output only. The email address of the user who created the
      template.
    description: Optional. Brief description of the template.
    environmentConfig: Optional. Environment configuration for session
      execution.
    jupyterSession: Optional. Jupyter session config.
    labels: Optional. Labels to associate with sessions created using this
      template. Label keys must contain 1 to 63 characters, and must conform
      to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt). Label values can be
      empty, but, if present, must contain 1 to 63 characters and conform to
      RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt). No more than 32 labels
      can be associated with a session.
    name: Required. The resource name of the session template.
    runtimeConfig: Optional. Runtime configuration for session execution.
    spark: Optional. Spark engine config.
    sparkConnectSession: Optional. Spark connect session config.
    updateTime: Output only. The time the template was last updated.
    uuid: Output only. A session template UUID (Unique Universal Identifier).
      The service generates this value when it creates the session template.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels to associate with sessions created using this
    template. Label keys must contain 1 to 63 characters, and must conform to
    RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt). Label values can be
    empty, but, if present, must contain 1 to 63 characters and conform to RFC
    1035 (https://www.ietf.org/rfc/rfc1035.txt). No more than 32 labels can be
    associated with a session.

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
    createTime = _messages.StringField(1)
    creator = _messages.StringField(2)
    description = _messages.StringField(3)
    environmentConfig = _messages.MessageField('EnvironmentConfig', 4)
    jupyterSession = _messages.MessageField('JupyterConfig', 5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    runtimeConfig = _messages.MessageField('RuntimeConfig', 8)
    spark = _messages.MessageField('SparkConfig', 9)
    sparkConnectSession = _messages.MessageField('SparkConnectConfig', 10)
    updateTime = _messages.StringField(11)
    uuid = _messages.StringField(12)