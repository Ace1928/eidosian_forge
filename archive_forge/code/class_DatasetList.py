from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetList(_messages.Message):
    """A DatasetList object.

  Messages:
    DatasetsValueListEntry: A DatasetsValueListEntry object.

  Fields:
    datasets: An array of the dataset resources in the project. Each resource
      contains basic information. For full information about a particular
      dataset resource, use the Datasets: get method. This property is omitted
      when there are no datasets in the project.
    etag: A hash value of the results page. You can use this property to
      determine if the page has changed since the last request.
    kind: The list type. This property always returns the value
      "bigquery#datasetList".
    nextPageToken: A token that can be used to request the next results page.
      This property is omitted on the final results page.
  """

    class DatasetsValueListEntry(_messages.Message):
        """A DatasetsValueListEntry object.

    Messages:
      LabelsValue: [Experimental] The labels associated with this dataset. You
        can use these to organize and group your datasets.

    Fields:
      datasetReference: The dataset reference. Use this property to access
        specific parts of the dataset's ID, such as project ID or dataset ID.
      friendlyName: A descriptive name for the dataset, if one exists.
      id: The fully-qualified, unique, opaque ID of the dataset.
      kind: The resource type. This property always returns the value
        "bigquery#dataset".
      labels: [Experimental] The labels associated with this dataset. You can
        use these to organize and group your datasets.
    """

        @encoding.MapUnrecognizedFields('additionalProperties')
        class LabelsValue(_messages.Message):
            """[Experimental] The labels associated with this dataset. You can use
      these to organize and group your datasets.

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
        datasetReference = _messages.MessageField('DatasetReference', 1)
        friendlyName = _messages.StringField(2)
        id = _messages.StringField(3)
        kind = _messages.StringField(4, default=u'bigquery#dataset')
        labels = _messages.MessageField('LabelsValue', 5)
    datasets = _messages.MessageField('DatasetsValueListEntry', 1, repeated=True)
    etag = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'bigquery#datasetList')
    nextPageToken = _messages.StringField(4)