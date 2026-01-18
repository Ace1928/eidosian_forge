from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsSpecsDeleteRevisionRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsSpecsDeleteRevisionRequest
  object.

  Fields:
    name: Required. The name of the spec revision to be deleted, with a
      revision ID explicitly included. Example: `projects/sample/locations/glo
      bal/apis/petstore/versions/1.0.0/specs/openapi.yaml@c7cfa2a8`
  """
    name = _messages.StringField(1, required=True)