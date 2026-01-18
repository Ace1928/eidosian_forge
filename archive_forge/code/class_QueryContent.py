from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryContent(_messages.Message):
    """The query content.

  Fields:
    iamPolicyAnalysisQuery: An IAM Policy Analysis query, which could be used
      in the AssetService.AnalyzeIamPolicy RPC or the
      AssetService.AnalyzeIamPolicyLongrunning RPC.
  """
    iamPolicyAnalysisQuery = _messages.MessageField('IamPolicyAnalysisQuery', 1)