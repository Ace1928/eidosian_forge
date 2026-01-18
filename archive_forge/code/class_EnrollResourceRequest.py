from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnrollResourceRequest(_messages.Message):
    """Request message to subscribe the Audit Manager service for given
  resource.

  Fields:
    destinations: Required. List of destination among which customer can
      choose to upload their reports during the audit process. While enrolling
      at a folder level, customer can choose Cloud storage bucket in any
      project. If the audit is triggered at project level using the service
      agent at folder level, all the destination options associated with
      folder level service agent will be available to auditing projects.
  """
    destinations = _messages.MessageField('EligibleDestination', 1, repeated=True)