from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV1ServiceConfig(_messages.Message):
    """The configuration of the service.

  Fields:
    apis: A list of API interfaces exported by this service. Contains only the
      names, versions, and method names of the interfaces.
    authentication: Auth configuration. Contains only the OAuth rules.
    documentation: Additional API documentation. Contains only the summary and
      the documentation URL.
    endpoints: Configuration for network endpoints. Contains only the names
      and aliases of the endpoints.
    name: The DNS address at which this service is available.  An example DNS
      address would be: `calendar.googleapis.com`.
    quota: Quota configuration.
    title: The product title for this service.
    usage: Configuration controlling usage of this service.
  """
    apis = _messages.MessageField('Api', 1, repeated=True)
    authentication = _messages.MessageField('Authentication', 2)
    documentation = _messages.MessageField('Documentation', 3)
    endpoints = _messages.MessageField('Endpoint', 4, repeated=True)
    name = _messages.StringField(5)
    quota = _messages.MessageField('Quota', 6)
    title = _messages.StringField(7)
    usage = _messages.MessageField('Usage', 8)