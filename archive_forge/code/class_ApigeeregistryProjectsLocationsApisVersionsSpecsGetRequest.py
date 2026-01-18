from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsSpecsGetRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsSpecsGetRequest object.

  Fields:
    name: Required. The name of the spec to retrieve. Format:
      `projects/*/locations/*/apis/*/versions/*/specs/*`
  """
    name = _messages.StringField(1, required=True)