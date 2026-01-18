from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresGetRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsAnnotationStoresGetRequest object.

  Fields:
    name: Required. The resource name of the Annotation store to get.
  """
    name = _messages.StringField(1, required=True)