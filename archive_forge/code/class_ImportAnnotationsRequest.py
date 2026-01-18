from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportAnnotationsRequest(_messages.Message):
    """Request to import Annotations. The Annotations to be imported must have
  client-supplied resource names which indicate the annotation resource. The
  import operation is not atomic. If a failure occurs, any annotations already
  imported are not removed.

  Fields:
    gcsSource: A GoogleCloudHealthcareV1alpha2AnnotationGcsSource attribute.
  """
    gcsSource = _messages.MessageField('GoogleCloudHealthcareV1alpha2AnnotationGcsSource', 1)