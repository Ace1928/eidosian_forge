from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class StandardResourceMetadata(_messages.Message):
    """The standard metadata of a cloud resource.

  Messages:
    LabelsValue: Labels associated with this resource. See [Labelling and
      grouping GCP
      resources](https://cloud.google.com/blog/products/gcp/labelling-and-
      grouping-your-google-cloud-platform-resources) for more information.

  Fields:
    additionalAttributes: Additional searchable attributes of this resource.
      Informational only. The exact set of attributes is subject to change.
      For example: project id, DNS name etc.
    assetType: The type of this resource. For example:
      "compute.googleapis.com/Disk".
    description: One or more paragraphs of text description of this resource.
      Maximum length could be up to 1M bytes.
    displayName: The display name of this resource.
    labels: Labels associated with this resource. See [Labelling and grouping
      GCP resources](https://cloud.google.com/blog/products/gcp/labelling-and-
      grouping-your-google-cloud-platform-resources) for more information.
    location: Location can be "global", regional like "us-east1", or zonal
      like "us-west1-b".
    name: The full resource name. For example: `//compute.googleapis.com/proje
      cts/my_project_123/zones/zone1/instances/instance1`. See [Resource Names
      ](https://cloud.google.com/apis/design/resource_names#full_resource_name
      ) for more information.
    networkTags: Network tags associated with this resource. Like labels,
      network tags are a type of annotations used to group GCP resources. See
      [Labelling GCP
      resources](lhttps://cloud.google.com/blog/products/gcp/labelling-and-
      grouping-your-google-cloud-platform-resources) for more information.
    project: The project that this resource belongs to, in the form of
      `projects/{project_number}`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with this resource. See [Labelling and grouping GCP
    resources](https://cloud.google.com/blog/products/gcp/labelling-and-
    grouping-your-google-cloud-platform-resources) for more information.

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
    additionalAttributes = _messages.StringField(1, repeated=True)
    assetType = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    location = _messages.StringField(6)
    name = _messages.StringField(7)
    networkTags = _messages.StringField(8, repeated=True)
    project = _messages.StringField(9)