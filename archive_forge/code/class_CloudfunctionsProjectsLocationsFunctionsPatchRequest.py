from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsPatchRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsPatchRequest object.

  Fields:
    function: A Function resource to be passed as the request body.
    name: A user-defined name of the function. Function names must be unique
      globally and match pattern `projects/*/locations/*/functions/*`
    updateMask: The list of fields to be updated. If no field mask is
      provided, all provided fields in the request will be updated.
  """
    function = _messages.MessageField('Function', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)