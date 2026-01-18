from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlossaryTermsPair(_messages.Message):
    """Represents a single entry for an unidirectional glossary.

  Fields:
    sourceTerm: The source term is the term that will get match in the text,
    targetTerm: The term that will replace the match source term.
  """
    sourceTerm = _messages.MessageField('GlossaryTerm', 1)
    targetTerm = _messages.MessageField('GlossaryTerm', 2)