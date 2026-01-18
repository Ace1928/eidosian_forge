from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1DenyAnalysisResultDenyDetail(_messages.Message):
    """A deny detail that explains which IAM deny rule denies the
  denied_access_tuple.

  Fields:
    accesses: The accesses that are denied. This could be the
      AccessTuple.access, or a subset of it. For example, if the
      AccessTuple.access is a role, this field could contain permissions in
      that role that are denied.
    denyRule: A deny rule in an IAM deny policy.
    exceptionIdentities: The identities that are exceptions from deny. This
      field is populated when: * The deny_rule has `exception_principals`; *
      For each exception_principal EP, EP is IN identities;
    identities: The identities that are denied. This could be the
      AccessTuple.identity, or its subset. For example, if the
      AccessTuple.identity is a group, this field could contain user accounts
      in that group that are denied. This field is populated with: * The
      [AccessTuple.identity] if it's IN the deny_rule's `denied_principals`,
      and not IN the `exception_principals`; * For each denied principal DP in
      the deny_rule's `denied_principals`, DP is s IN the
      [AccessTuple.identity] and not IN the `exception_principals`; The IN
      operator is defined as below: * An identity is in an identities list,
      e.g.: user:foo@ in [user:foo@, user:bar@, group:baz@]; * An identity is
      in a member of an identity of a list, e.g.: user:foo@ is a member of
      group:baz@, which is in a list [user:bar@, group:baz@];
    resources: The resources that are denied. This could be the
      AccessTuple.resource, or its descendant resources. For example, if the
      AccessTuple.resource is a project, this field could contain BigQuery
      datasets in that project that are denied.
  """
    accesses = _messages.MessageField('GoogleCloudAssetV1DenyAnalysisResultAccess', 1, repeated=True)
    denyRule = _messages.MessageField('GoogleIamV2DenyRule', 2)
    exceptionIdentities = _messages.MessageField('GoogleCloudAssetV1DenyAnalysisResultIdentity', 3, repeated=True)
    identities = _messages.MessageField('GoogleCloudAssetV1DenyAnalysisResultIdentity', 4, repeated=True)
    resources = _messages.MessageField('GoogleCloudAssetV1DenyAnalysisResultResource', 5, repeated=True)