from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsWaitersCreateRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsWaitersCreateRequest object.

  Fields:
    parent: The path to the configuration that will own the waiter. The
      configuration must exist beforehand; the path must be in the format:
      `projects/[PROJECT_ID]/configs/[CONFIG_NAME]`.
    requestId: An optional but recommended unique `request_id`. If the server
      receives two `create()` requests with the same `request_id`, then the
      second request will be ignored and the first resource created and stored
      in the backend is returned. Empty `request_id` fields are ignored. It is
      responsibility of the client to ensure uniqueness of the `request_id`
      strings. `request_id` strings are limited to 64 characters.
    waiter: A Waiter resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    waiter = _messages.MessageField('Waiter', 3)