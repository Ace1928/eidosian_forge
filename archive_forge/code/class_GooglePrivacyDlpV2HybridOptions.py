from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2HybridOptions(_messages.Message):
    """Configuration to control jobs where the content being inspected is
  outside of Google Cloud Platform.

  Messages:
    LabelsValue: To organize findings, these labels will be added to each
      finding. Label keys must be between 1 and 63 characters long and must
      conform to the following regular expression:
      `[a-z]([-a-z0-9]*[a-z0-9])?`. Label values must be between 0 and 63
      characters long and must conform to the regular expression
      `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10 labels can be
      associated with a given finding. Examples: * `"environment" :
      "production"` * `"pipeline" : "etl"`

  Fields:
    description: A short description of where the data is coming from. Will be
      stored once in the job. 256 max length.
    labels: To organize findings, these labels will be added to each finding.
      Label keys must be between 1 and 63 characters long and must conform to
      the following regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. Label
      values must be between 0 and 63 characters long and must conform to the
      regular expression `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10
      labels can be associated with a given finding. Examples: *
      `"environment" : "production"` * `"pipeline" : "etl"`
    requiredFindingLabelKeys: These are labels that each inspection request
      must include within their 'finding_labels' map. Request may contain
      others, but any missing one of these will be rejected. Label keys must
      be between 1 and 63 characters long and must conform to the following
      regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. No more than 10 keys
      can be required.
    tableOptions: If the container is a table, additional information to make
      findings meaningful such as the columns that are primary keys.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """To organize findings, these labels will be added to each finding.
    Label keys must be between 1 and 63 characters long and must conform to
    the following regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. Label
    values must be between 0 and 63 characters long and must conform to the
    regular expression `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10 labels
    can be associated with a given finding. Examples: * `"environment" :
    "production"` * `"pipeline" : "etl"`

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
    description = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    requiredFindingLabelKeys = _messages.StringField(3, repeated=True)
    tableOptions = _messages.MessageField('GooglePrivacyDlpV2TableOptions', 4)