from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsCreateRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsCreateRequest
  object.

  Fields:
    annotation: A Annotation resource to be passed as the request body.
    parent: Required. The name of the Annotation store this annotation belongs
      to. For example, `projects/my-project/locations/us-
      central1/datasets/mydataset/annotationStores/myannotationstore`.
  """
    annotation = _messages.MessageField('Annotation', 1)
    parent = _messages.StringField(2, required=True)