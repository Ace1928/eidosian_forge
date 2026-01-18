from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2HybridFindingDetails(_messages.Message):
    """Populate to associate additional data with each finding.

  Messages:
    LabelsValue: Labels to represent user provided metadata about the data
      being inspected. If configured by the job, some key values may be
      required. The labels associated with `Finding`'s produced by hybrid
      inspection. Label keys must be between 1 and 63 characters long and must
      conform to the following regular expression:
      `[a-z]([-a-z0-9]*[a-z0-9])?`. Label values must be between 0 and 63
      characters long and must conform to the regular expression
      `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10 labels can be
      associated with a given finding. Examples: * `"environment" :
      "production"` * `"pipeline" : "etl"`

  Fields:
    containerDetails: Details about the container where the content being
      inspected is from.
    fileOffset: Offset in bytes of the line, from the beginning of the file,
      where the finding is located. Populate if the item being scanned is only
      part of a bigger item, such as a shard of a file and you want to track
      the absolute position of the finding.
    labels: Labels to represent user provided metadata about the data being
      inspected. If configured by the job, some key values may be required.
      The labels associated with `Finding`'s produced by hybrid inspection.
      Label keys must be between 1 and 63 characters long and must conform to
      the following regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. Label
      values must be between 0 and 63 characters long and must conform to the
      regular expression `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10
      labels can be associated with a given finding. Examples: *
      `"environment" : "production"` * `"pipeline" : "etl"`
    rowOffset: Offset of the row for tables. Populate if the row(s) being
      scanned are part of a bigger dataset and you want to keep track of their
      absolute position.
    tableOptions: If the container is a table, additional information to make
      findings meaningful such as the columns that are primary keys. If not
      known ahead of time, can also be set within each inspect hybrid call and
      the two will be merged. Note that identifying_fields will only be stored
      to BigQuery, and only if the BigQuery action has been included.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels to represent user provided metadata about the data being
    inspected. If configured by the job, some key values may be required. The
    labels associated with `Finding`'s produced by hybrid inspection. Label
    keys must be between 1 and 63 characters long and must conform to the
    following regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. Label values
    must be between 0 and 63 characters long and must conform to the regular
    expression `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10 labels can be
    associated with a given finding. Examples: * `"environment" :
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
    containerDetails = _messages.MessageField('GooglePrivacyDlpV2Container', 1)
    fileOffset = _messages.IntegerField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    rowOffset = _messages.IntegerField(4)
    tableOptions = _messages.MessageField('GooglePrivacyDlpV2TableOptions', 5)