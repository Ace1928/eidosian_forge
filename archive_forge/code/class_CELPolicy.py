from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CELPolicy(_messages.Message):
    """Authorization policy schema. A policy is composed of a set of Rules that
  specifies the conditions that need to be met in order for an authorization
  decision to be finalized. For example, a policy could specify that all
  requests to a specific IP address or that requests from a specific SPIFFE ID
  must be allowed. Policy can also contain a CEL expression for custom checks.

  Fields:
    ruleBlocks: List of rule blocks.
  """
    ruleBlocks = _messages.MessageField('RuleBlock', 1, repeated=True)