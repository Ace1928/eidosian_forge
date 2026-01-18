from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainConfig(_messages.Message):
    """A DomainConfig object.

  Fields:
    domain: Domain name for this config.
    routes: A list of route configurations to associate with the domain. Each
      Route configuration must include a paths configuration.
  """
    domain = _messages.StringField(1)
    routes = _messages.MessageField('Route', 2, repeated=True)