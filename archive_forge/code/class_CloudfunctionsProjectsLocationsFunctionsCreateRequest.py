from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsCreateRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsCreateRequest object.

  Fields:
    function: A Function resource to be passed as the request body.
    functionId: The ID to use for the function, which will become the final
      component of the function's resource name. This value should be 4-63
      characters, and valid characters are /a-z-/.
    parent: Required. The project and location in which the function should be
      created, specified in the format `projects/*/locations/*`
  """
    function = _messages.MessageField('Function', 1)
    functionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)