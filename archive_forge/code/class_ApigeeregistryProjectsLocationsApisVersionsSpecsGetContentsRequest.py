from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsSpecsGetContentsRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsSpecsGetContentsRequest
  object.

  Fields:
    name: Required. The name of the spec whose contents should be retrieved.
      Format: `projects/*/locations/*/apis/*/versions/*/specs/*`
  """
    name = _messages.StringField(1, required=True)