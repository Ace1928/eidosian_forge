from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamSearchApplicablePoliciesSearchRequest(_messages.Message):
    """A IamSearchApplicablePoliciesSearchRequest object.

  Fields:
    filter: Optional. Filtering currently only supports the kind of policies
      to return, and must be in the format "kind:[policyKind1] OR
      kind:[policyKind2]". New policy kinds may be added in the future without
      notice. Example value: "kind:principalAccessBoundaryPolicies"
    pageSize: Optional. The limit of number of items (binding+policy pairs) to
      return. The default and maximum is 100 and values above 100 are
      truncated to 100.
    pageToken: Optional. A page token, received from a previous
      `SearchApplicablePolicies` call.
    targetQuery: Required. The target for which to list the policies and
      bindings for. Binding conditions will not be evaluated and all bindings
      that are bound to the target will be returned. All targets from the
      CreatePolicyBinding request are supported, as well as principals that
      are part of the principalSet. e.g.
      principalSet://iam.googleapis.com/projects/1234/*
      principal:alice@acme.com
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    targetQuery = _messages.StringField(4)