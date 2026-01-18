from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesCreateRequest(_messages.Message):
    """A RunProjectsLocationsServicesCreateRequest object.

  Fields:
    googleCloudRunV2Service: A GoogleCloudRunV2Service resource to be passed
      as the request body.
    parent: Required. The location and project in which this service should be
      created. Format: projects/{project}/locations/{location}, where
      {project} can be project id or number. Only lowercase characters,
      digits, and hyphens.
    serviceId: Required. The unique identifier for the Service. It must begin
      with letter, and cannot end with hyphen; must contain fewer than 50
      characters. The name of the service becomes
      {parent}/services/{service_id}.
    validateOnly: Indicates that the request should be validated and default
      values populated, without persisting the request or creating any
      resources.
  """
    googleCloudRunV2Service = _messages.MessageField('GoogleCloudRunV2Service', 1)
    parent = _messages.StringField(2, required=True)
    serviceId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)