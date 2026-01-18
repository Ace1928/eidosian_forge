from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeidentifyConfig(_messages.Message):
    """Configures de-id options specific to different types of content. Each
  submessage customizes the handling of an https://tools.ietf.org/html/rfc6838
  media type or subtype. Configs are applied in a nested manner at runtime.

  Fields:
    annotation: Configures how annotations (such as the location and infoTypes
      of sensitive information) are created during de-identification. If
      unspecified, no annotations are created.
    dicom: Configures de-id of application/DICOM content. Deprecated. Use
      `dicom_tag_config` instead.
    fhir: Configures de-id of application/FHIR content. Deprecated. Use
      `fhir_field_config` instead.
    image: Configures the de-identification of image pixels in the
      source_dataset. Deprecated. Use `dicom_tag_config.options.clean_image`
      instead.
    text: Configures the de-identification of text in `source_dataset`.
  """
    annotation = _messages.MessageField('AnnotationConfig', 1)
    dicom = _messages.MessageField('DicomConfig', 2)
    fhir = _messages.MessageField('FhirConfig', 3)
    image = _messages.MessageField('ImageConfig', 4)
    text = _messages.MessageField('TextConfig', 5)