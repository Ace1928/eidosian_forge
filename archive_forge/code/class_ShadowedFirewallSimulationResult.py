from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShadowedFirewallSimulationResult(_messages.Message):
    """ShadowedFirewallSimulationResult contains results for shadowed firewall
  analysis from two network configurations, i.e. original and proposed network
  configurations.

  Fields:
    baseConfigResults: Results for simulating shadowed firewall analysis with
      original network configurations. There can be more than one shadowing
      relation for a particular shadowing firewall.
    firewall: Relative resource path (i.e. uri) of the firewall rule that is
      getting shadowed:
      'projects/{project_id}/{location}/firewalls/{firewall_name}'
    network: Relative resource path (i.e. uri) of the network in which the
      shadowed firewall belongs:
      'projects/{project_id}/{location}/networks/{network_name}'
    proposedConfigResults: Results for simulating shadowed firewall analysis
      with proposed network configurations. There can be more than one
      shadowing relation for a particular shadowing firewall.
    resultsDiffer: Indicates if the results from running shadowed firewall
      analysis with the original network configurations and with the proposed
      network configurations differ.
  """
    baseConfigResults = _messages.MessageField('ShadowingInfo', 1, repeated=True)
    firewall = _messages.StringField(2)
    network = _messages.StringField(3)
    proposedConfigResults = _messages.MessageField('ShadowingInfo', 4, repeated=True)
    resultsDiffer = _messages.BooleanField(5)