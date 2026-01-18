from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExtensionChain(_messages.Message):
    """A single extension chain wrapper that contains the match conditions and
  extensions to execute.

  Fields:
    extensions: Required. A set of extensions to execute for the matching
      request. At least one extension is required. Up to 3 extensions can be
      defined for each extension chain for `LbTrafficExtension` resource.
      `LbRouteExtension` chains are limited to 1 extension per extension
      chain.
    matchCondition: Required. Conditions under which this chain is invoked for
      a request.
    name: Required. The name for this extension chain. The name is logged as
      part of the HTTP request logs. The name must conform with RFC-1034, is
      restricted to lower-cased letters, numbers and hyphens, and can have a
      maximum length of 63 characters. Additionally, the first character must
      be a letter and the last a letter or a number.
  """
    extensions = _messages.MessageField('ExtensionChainExtension', 1, repeated=True)
    matchCondition = _messages.MessageField('ExtensionChainMatchCondition', 2)
    name = _messages.StringField(3)