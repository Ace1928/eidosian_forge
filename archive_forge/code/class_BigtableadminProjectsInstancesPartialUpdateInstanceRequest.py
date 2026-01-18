from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesPartialUpdateInstanceRequest(_messages.Message):
    """A BigtableadminProjectsInstancesPartialUpdateInstanceRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: The unique name of the instance. Values are of the form
      `projects/{project}/instances/a-z+[a-z0-9]`.
    updateMask: Required. The subset of Instance fields which should be
      replaced. Must be explicitly set.
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)