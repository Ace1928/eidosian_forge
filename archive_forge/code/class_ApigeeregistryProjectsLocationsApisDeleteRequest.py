from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeleteRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeleteRequest object.

  Fields:
    force: If set to true, any child resources will also be deleted.
      (Otherwise, the request will only work if there are no child resources.)
    name: Required. The name of the API to delete. Format:
      `projects/*/locations/*/apis/*`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)