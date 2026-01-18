from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResponsePolicyRule(_messages.Message):
    """A Response Policy Rule is a selector that applies its behavior to
  queries that match the selector. Selectors are DNS names, which may be
  wildcards or exact matches. Each DNS query subject to a Response Policy
  matches at most one ResponsePolicyRule, as identified by the dns_name field
  with the longest matching suffix.

  Enums:
    BehaviorValueValuesEnum: Answer this query with a behavior rather than DNS
      data.

  Fields:
    behavior: Answer this query with a behavior rather than DNS data.
    dnsName: The DNS name (wildcard or exact) to apply this rule to. Must be
      unique within the Response Policy Rule.
    kind: A string attribute.
    localData: Answer this query directly with DNS data. These
      ResourceRecordSets override any other DNS behavior for the matched name;
      in particular they override private zones, the public internet, and GCP
      internal DNS. No SOA nor NS types are allowed.
    ruleName: An identifier for this rule. Must be unique with the
      ResponsePolicy.
  """

    class BehaviorValueValuesEnum(_messages.Enum):
        """Answer this query with a behavior rather than DNS data.

    Values:
      behaviorUnspecified: <no description>
      bypassResponsePolicy: Skip a less-specific ResponsePolicyRule and
        continue normal query logic. This can be used with a less-specific
        wildcard selector to exempt a subset of the wildcard
        ResponsePolicyRule from the ResponsePolicy behavior and query the
        public Internet instead. For instance, if these rules exist:
        *.example.com -> LocalData 1.2.3.4 foo.example.com -> Behavior
        'bypassResponsePolicy' Then a query for 'foo.example.com' skips the
        wildcard. This additionally functions to facilitate the allowlist
        feature. RPZs can be applied to multiple levels in the (eventually
        org, folder, project, network) hierarchy. If a rule is applied at a
        higher level of the hierarchy, adding a passthru rule at a lower level
        will supersede that, and a query from an affected vm to that domain
        will be exempt from the RPZ and proceed to normal resolution behavior.
    """
        behaviorUnspecified = 0
        bypassResponsePolicy = 1
    behavior = _messages.EnumField('BehaviorValueValuesEnum', 1)
    dnsName = _messages.StringField(2)
    kind = _messages.StringField(3, default='dns#responsePolicyRule')
    localData = _messages.MessageField('ResponsePolicyRuleLocalData', 4)
    ruleName = _messages.StringField(5)