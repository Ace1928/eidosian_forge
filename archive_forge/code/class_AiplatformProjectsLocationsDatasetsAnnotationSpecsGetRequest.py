from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsAnnotationSpecsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsAnnotationSpecsGetRequest object.

  Fields:
    name: Required. The name of the AnnotationSpec resource. Format: `projects
      /{project}/locations/{location}/datasets/{dataset}/annotationSpecs/{anno
      tation_spec}`
    readMask: Mask specifying which fields to read.
  """
    name = _messages.StringField(1, required=True)
    readMask = _messages.StringField(2)