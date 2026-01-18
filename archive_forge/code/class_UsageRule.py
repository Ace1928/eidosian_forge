from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsageRule(_messages.Message):
    """Usage configuration rules for the service.  NOTE: Under development.
  Use this rule to configure unregistered calls for the service. Unregistered
  calls are calls that do not contain consumer project identity. (Example:
  calls that do not contain an API key). By default, API methods do not allow
  unregistered calls, and each method call must be identified by a consumer
  project identity. Use this rule to allow/disallow unregistered calls.
  Example of an API that wants to allow unregistered calls for entire service.
  usage:       rules:       - selector: "*"         allow_unregistered_calls:
  true  Example of a method that wants to allow unregistered calls.
  usage:       rules:       - selector:
  "google.example.library.v1.LibraryService.CreateBook"
  allow_unregistered_calls: true

  Fields:
    allowUnregisteredCalls: True, if the method allows unregistered calls;
      false otherwise.
    selector: Selects the methods to which this rule applies. Use '*' to
      indicate all methods in all APIs.  Refer to selector for syntax details.
  """
    allowUnregisteredCalls = _messages.BooleanField(1)
    selector = _messages.StringField(2)