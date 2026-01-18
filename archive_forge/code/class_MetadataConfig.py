from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataConfig(_messages.Message):
    """A MetadataConfig object.

  Fields:
    complexType: Required. Reference to the complex type name, in the
      following form:
      `projects/{project}/locations/{location}/complexTypes/{name}`.
    moduleDefinition: Reference to the module definition name, in the
      following form:
      `projects/{project}/locations/{location}/modules/{name}#{definition}`.
    owner: Output only. The owner of the metadata, set by the system.
    required: If true, this asset metadata is required to be specified during
      asset creation.
  """
    complexType = _messages.StringField(1)
    moduleDefinition = _messages.StringField(2)
    owner = _messages.StringField(3)
    required = _messages.BooleanField(4)