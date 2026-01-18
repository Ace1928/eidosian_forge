from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductSetsImportRequest(_messages.Message):
    """A VisionProjectsLocationsProductSetsImportRequest object.

  Fields:
    importProductSetsRequest: A ImportProductSetsRequest resource to be passed
      as the request body.
    parent: Required. The project in which the ProductSets should be imported.
      Format is `projects/PROJECT_ID/locations/LOC_ID`.
  """
    importProductSetsRequest = _messages.MessageField('ImportProductSetsRequest', 1)
    parent = _messages.StringField(2, required=True)