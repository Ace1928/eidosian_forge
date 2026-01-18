from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisibilityRule(_messages.Message):
    """A visibility rule provides visibility configuration for an individual
  API element.

  Fields:
    enforceRuntimeVisibility: Controls whether visibility is enforced at
      runtime for requests to an API method. This setting has meaning only
      when the selector applies to a method or an API.  If true, requests
      without method visibility will receive a NOT_FOUND error, and any non-
      visible fields will be scrubbed from the response messages. The default
      is determined by the value of
      google.api.Visibility.enforce_runtime_visibility.
    restriction: Lists the visibility labels for this rule. Any of the listed
      labels grants visibility to the element.  If a rule has multiple labels,
      removing one of the labels but not all of them can break clients.
      Example:      visibility:       rules:       - selector:
      google.calendar.Calendar.EnhancedSearch         restriction:
      GOOGLE_INTERNAL, TRUSTED_TESTER  Removing GOOGLE_INTERNAL from this
      restriction will break clients that rely on this method and only had
      access to it through GOOGLE_INTERNAL.
    selector: Selects methods, messages, fields, enums, etc. to which this
      rule applies.  Refer to selector for syntax details.
  """
    enforceRuntimeVisibility = _messages.BooleanField(1)
    restriction = _messages.StringField(2)
    selector = _messages.StringField(3)