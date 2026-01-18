from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsInstancesCreateRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsInstancesCreateRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    instanceId: Required. Identifier to assign to the Instance. Must be unique
      within scope of the parent resource.
    parent: Required. Parent resource of the Instance, of the form:
      `projects/*/locations/*`
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)