from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContextualDeidConfig(_messages.Message):
    """Fields that don't match a KeepField or CleanTextField `action` in the
  BASIC profile are collected into a contextual phrase list. For fields that
  match a CleanTextField `action` in FieldMetadata or ProfileType, the process
  attempts to transform phrases matching these contextual entries. These
  contextual phrases are replaced with the token "[CTX]". This feature uses an
  additional InfoType during inspection.
  """