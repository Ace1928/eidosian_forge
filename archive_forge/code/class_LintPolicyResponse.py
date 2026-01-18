from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LintPolicyResponse(_messages.Message):
    """The response of a lint operation. An empty response indicates the
  operation was able to fully execute and no lint issue was found.

  Fields:
    lintResults: List of lint results sorted by `severity` in descending
      order.
  """
    lintResults = _messages.MessageField('LintResult', 1, repeated=True)