from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsApisConfigsDeleteRequest(_messages.Message):
    """A ApigatewayProjectsLocationsApisConfigsDeleteRequest object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/global/apis/*/configs/*`
  """
    name = _messages.StringField(1, required=True)