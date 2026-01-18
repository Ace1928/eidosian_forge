from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaSearchApplicablePoliciesResponse(_messages.Message):
    """Response message for SearchApplicablePolicies

  Fields:
    bindingsAndPolicies: A list of Bindings and the policies associated with
      those bindings The bindings will be ordered by enforcement point
      starting from the lowest at the target level and up the CRM hierarchy.
      No order is guaranteed for bindings for a given enforcement point.
    nextPageToken: The page token to use in a follow up
      SearchApplicablePolicies request
    responseComplete: Does the response contain the full list of all bindings
      and policies applicable or were some excluded due to lack of permissions
  """
    bindingsAndPolicies = _messages.MessageField('GoogleIamV3betaSearchApplicablePoliciesResponseBindingAndPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    responseComplete = _messages.BooleanField(3)