from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsCatalogsGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsCatalogsGetRequest object.

  Fields:
    name: Required. The name of the catalog to retrieve, in the following
      form: `projects/{project}/locations/{location}/catalogs/{catalog}`.
  """
    name = _messages.StringField(1, required=True)