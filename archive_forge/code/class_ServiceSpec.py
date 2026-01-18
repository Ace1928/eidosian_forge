from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceSpec(_messages.Message):
    """ServiceSpec holds the desired state of the Route (from the client),
  which is used to manipulate the underlying Route and Configuration(s).

  Fields:
    template: Holds the latest specification for the Revision to be stamped
      out.
    traffic: Specifies how to distribute traffic over a collection of Knative
      Revisions and Configurations to the Service's main URL.
  """
    template = _messages.MessageField('RevisionTemplate', 1)
    traffic = _messages.MessageField('TrafficTarget', 2, repeated=True)