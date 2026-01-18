from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShadowingInfo(_messages.Message):
    """ShadowingInfo defines a list of firewalls that are causing shadowing for
  a particular firewall.

  Fields:
    shadowingFirewalls: Relative resource path (i.e. uri) of the resource that
      is causing shadowing:
      'projects/{project_id}/{location}/firewalls/{firewall_name}'
  """
    shadowingFirewalls = _messages.StringField(1, repeated=True)