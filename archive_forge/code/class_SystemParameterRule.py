from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SystemParameterRule(_messages.Message):
    """Define a system parameter rule mapping system parameter definitions to
  methods.

  Fields:
    parameters: Define parameters. Multiple names may be defined for a
      parameter. For a given method call, only one of them should be used. If
      multiple names are used the behavior is implementation-dependent. If
      none of the specified names are present the behavior is parameter-
      dependent.
    selector: Selects the methods to which this rule applies. Use '*' to
      indicate all methods in all APIs.  Refer to selector for syntax details.
  """
    parameters = _messages.MessageField('SystemParameter', 1, repeated=True)
    selector = _messages.StringField(2)