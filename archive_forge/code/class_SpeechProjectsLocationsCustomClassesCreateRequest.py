from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsCustomClassesCreateRequest(_messages.Message):
    """A SpeechProjectsLocationsCustomClassesCreateRequest object.

  Fields:
    customClass: A CustomClass resource to be passed as the request body.
    customClassId: The ID to use for the CustomClass, which will become the
      final component of the CustomClass's resource name. This value should be
      4-63 characters, and valid characters are /a-z-/.
    parent: Required. The project and location where this CustomClass will be
      created. The expected format is
      `projects/{project}/locations/{location}`.
    validateOnly: If set, validate the request and preview the CustomClass,
      but do not actually create it.
  """
    customClass = _messages.MessageField('CustomClass', 1)
    customClassId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)