from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsModelsVersionsDeleteRequest(_messages.Message):
    """A MlProjectsModelsVersionsDeleteRequest object.

  Fields:
    name: Required. The name of the version. You can get the names of all the
      versions of a model by calling projects.models.versions.list.
  """
    name = _messages.StringField(1, required=True)