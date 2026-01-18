from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersUsersGetRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersUsersGetRequest object.

  Fields:
    name: Required. The name of the resource. For the required format, see the
      comment on the User.name field.
  """
    name = _messages.StringField(1, required=True)