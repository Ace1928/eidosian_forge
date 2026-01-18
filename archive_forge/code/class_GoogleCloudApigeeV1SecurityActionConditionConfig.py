from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityActionConditionConfig(_messages.Message):
    """The following are a list of conditions. A valid SecurityAction must
  contain at least one condition. Within a condition, each element is ORed.
  Across conditions elements are ANDed. For example if a SecurityAction has
  the following: ip_address_ranges: ["ip1", "ip2"] and bot_reasons:
  ["Flooder", "Robot Abuser"] then this is interpreted as: enforce the action
  if the incoming request has ((ip_address_ranges = "ip1" OR ip_address_ranges
  = "ip2") AND (bot_reasons="Flooder" OR bot_reasons="Robot Abuser")).
  Conditions other than ip_address_ranges and bot_reasons cannot be ANDed.

  Fields:
    accessTokens: Optional. A list of access_tokens. Limit 1000 per action.
    apiKeys: Optional. A list of API keys. Limit 1000 per action.
    apiProducts: Optional. A list of API Products. Limit 1000 per action.
    botReasons: Optional. A list of Bot Reasons. Current options: Flooder,
      Brute Guessor, Static Content Scraper, OAuth Abuser, Robot Abuser,
      TorListRule, Advanced Anomaly Detection, Advanced API Scraper, Search
      Engine Crawlers, Public Clouds, Public Cloud AWS, Public Cloud Azure,
      and Public Cloud Google.
    developerApps: Optional. A list of developer apps. Limit 1000 per action.
    developers: Optional. A list of developers. Limit 1000 per action.
    ipAddressRanges: Optional. A list of IP addresses. This could be either
      IPv4 or IPv6. Limited to 100 per action.
    userAgents: Optional. A list of user agents to deny. We look for exact
      matches. Limit 50 per action.
  """
    accessTokens = _messages.StringField(1, repeated=True)
    apiKeys = _messages.StringField(2, repeated=True)
    apiProducts = _messages.StringField(3, repeated=True)
    botReasons = _messages.StringField(4, repeated=True)
    developerApps = _messages.StringField(5, repeated=True)
    developers = _messages.StringField(6, repeated=True)
    ipAddressRanges = _messages.StringField(7, repeated=True)
    userAgents = _messages.StringField(8, repeated=True)