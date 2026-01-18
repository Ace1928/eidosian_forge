from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesInsertRequest(_messages.Message):
    """A ComputeInstancesInsertRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    project: Project ID for this request.
    requestId: An optional request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. For example,
      consider a situation where you make an initial request and the request
      times out. If you make the request again with the same request ID, the
      server can check if original operation with the same request ID was
      received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      ( 00000000-0000-0000-0000-000000000000).
    sourceInstanceTemplate: Specifies instance template to create the
      instance. This field is optional. It can be a full or partial URL. For
      example, the following are all valid URLs to an instance template: -
      https://www.googleapis.com/compute/v1/projects/project
      /global/instanceTemplates/instanceTemplate -
      projects/project/global/instanceTemplates/instanceTemplate -
      global/instanceTemplates/instanceTemplate
    sourceMachineImage: Specifies the machine image to use to create the
      instance. This field is optional. It can be a full or partial URL. For
      example, the following are all valid URLs to a machine image: -
      https://www.googleapis.com/compute/v1/projects/project/global/global
      /machineImages/machineImage -
      projects/project/global/global/machineImages/machineImage -
      global/machineImages/machineImage
    zone: The name of the zone for this request.
  """
    instance = _messages.MessageField('Instance', 1)
    project = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    sourceInstanceTemplate = _messages.StringField(4)
    sourceMachineImage = _messages.StringField(5)
    zone = _messages.StringField(6, required=True)