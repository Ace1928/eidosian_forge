from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeIamPolicyResponse(_messages.Message):
    """A response message for AssetService.AnalyzeIamPolicy.

  Fields:
    fullyExplored: Represents whether all entries in the main_analysis and
      service_account_impersonation_analysis have been fully explored to
      answer the query in the request.
    mainAnalysis: The main analysis that matches the original request.
    serviceAccountImpersonationAnalysis: The service account impersonation
      analysis if
      AnalyzeIamPolicyRequest.analyze_service_account_impersonation is
      enabled.
  """
    fullyExplored = _messages.BooleanField(1)
    mainAnalysis = _messages.MessageField('IamPolicyAnalysis', 2)
    serviceAccountImpersonationAnalysis = _messages.MessageField('IamPolicyAnalysis', 3, repeated=True)