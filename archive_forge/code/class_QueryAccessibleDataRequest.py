from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryAccessibleDataRequest(_messages.Message):
    """Queries all data_ids that are consented for a given use in the given
  consent store and writes them to a specified destination. The returned
  Operation includes a progress counter for the number of User data mappings
  processed. Errors are logged to Cloud Logging (see [Viewing error logs in
  Cloud Logging] (https://cloud.google.com/healthcare/docs/how-tos/logging)
  and [QueryAccessibleData] for a sample log entry).

  Messages:
    RequestAttributesValue: The values of request attributes associated with
      this access request.
    ResourceAttributesValue: Optional. The values of resource attributes
      associated with the type of resources being requested. If no values are
      specified, then all resource types are included in the output.

  Fields:
    gcsDestination: The Cloud Storage destination. The Cloud Healthcare API
      service account must have the `roles/storage.objectAdmin` Cloud IAM role
      for this Cloud Storage location. The object name is in the following
      format: query-accessible-data-result-{operation_id}.txt where each line
      contains a single data_id.
    requestAttributes: The values of request attributes associated with this
      access request.
    resourceAttributes: Optional. The values of resource attributes associated
      with the type of resources being requested. If no values are specified,
      then all resource types are included in the output.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RequestAttributesValue(_messages.Message):
        """The values of request attributes associated with this access request.

    Messages:
      AdditionalProperty: An additional property for a RequestAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        RequestAttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RequestAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceAttributesValue(_messages.Message):
        """Optional. The values of resource attributes associated with the type
    of resources being requested. If no values are specified, then all
    resource types are included in the output.

    Messages:
      AdditionalProperty: An additional property for a ResourceAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ResourceAttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    gcsDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2ConsentGcsDestination', 1)
    requestAttributes = _messages.MessageField('RequestAttributesValue', 2)
    resourceAttributes = _messages.MessageField('ResourceAttributesValue', 3)