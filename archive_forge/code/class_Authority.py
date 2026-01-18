from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Authority(_messages.Message):
    """Note kind that represents a logical attestation "role" or "authority".
  For example, an organization might have one `Authority` for "QA" and one for
  "build". This note is intended to act strictly as a grouping mechanism for
  the attached occurrences (Attestations). This grouping mechanism also
  provides a security boundary, since IAM ACLs gate the ability for a
  principle to attach an occurrence to a given note. It also provides a single
  point of lookup to find all attached attestation occurrences, even if they
  don't all live in the same project.

  Fields:
    hint: Hint hints at the purpose of the attestation authority.
  """
    hint = _messages.MessageField('Hint', 1)