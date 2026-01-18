from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsFederationsGetRequest(_messages.Message):
    """A MetastoreProjectsLocationsFederationsGetRequest object.

  Fields:
    name: Required. The relative resource name of the metastore federation to
      retrieve, in the following form:projects/{project_number}/locations/{loc
      ation_id}/federations/{federation_id}.
  """
    name = _messages.StringField(1, required=True)