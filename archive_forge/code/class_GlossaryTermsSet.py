from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlossaryTermsSet(_messages.Message):
    """Represents a single entry for an equivalent term set glossary. This is
  used for equivalent term sets where each term can be replaced by the other
  terms in the set.

  Fields:
    terms: Each term in the set represents a term that can be replaced by the
      other terms.
  """
    terms = _messages.MessageField('GlossaryTerm', 1, repeated=True)