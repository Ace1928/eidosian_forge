from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintTemplateMetadata(_messages.Message):
    """MembershipConstraintTemplateMetadata contains relevant fields from
  constraint template metadata.

  Fields:
    creation: metadata.creation_timestamp from the constraint template.
    generation: metadata.generation from the constraint template.
  """
    creation = _messages.StringField(1)
    generation = _messages.IntegerField(2)