from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1EnvironmentGroupConfig(_messages.Message):
    """EnvironmentGroupConfig is a revisioned snapshot of an EnvironmentGroup
  and its associated routing rules.

  Fields:
    endpointChainingRules: A list of proxies in each deployment group for
      proxy chaining calls.
    hostnames: Host names for the environment group.
    location: When this message appears in the top-level IngressConfig, this
      field will be populated in lieu of the inlined routing_rules and
      hostnames fields. Some URL for downloading the full
      EnvironmentGroupConfig for this group.
    name: Name of the environment group in the following format:
      `organizations/{org}/envgroups/{envgroup}`.
    revisionId: Revision id that defines the ordering of the
      EnvironmentGroupConfig resource. The higher the revision, the more
      recently the configuration was deployed.
    routingRules: Ordered list of routing rules defining how traffic to this
      environment group's hostnames should be routed to different
      environments.
    uid: A unique id for the environment group config that will only change if
      the environment group is deleted and recreated.
  """
    endpointChainingRules = _messages.MessageField('GoogleCloudApigeeV1EndpointChainingRule', 1, repeated=True)
    hostnames = _messages.StringField(2, repeated=True)
    location = _messages.StringField(3)
    name = _messages.StringField(4)
    revisionId = _messages.IntegerField(5)
    routingRules = _messages.MessageField('GoogleCloudApigeeV1RoutingRule', 6, repeated=True)
    uid = _messages.StringField(7)