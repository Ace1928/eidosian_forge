from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeIamPolicyLongrunningRequest(_messages.Message):
    """A request message for AssetService.AnalyzeIamPolicyLongrunning.

  Fields:
    analysisQuery: Required. The request query.
    outputConfig: Required. Output configuration indicating where the results
      will be output to.
    savedAnalysisQuery: Optional. The name of a saved query, which must be in
      the format of: * projects/project_number/savedQueries/saved_query_id *
      folders/folder_number/savedQueries/saved_query_id *
      organizations/organization_number/savedQueries/saved_query_id If both
      `analysis_query` and `saved_analysis_query` are provided, they will be
      merged together with the `saved_analysis_query` as base and the
      `analysis_query` as overrides. For more details of the merge behavior,
      refer to the [MergeFrom](https://developers.google.com/protocol-buffers/
      docs/reference/cpp/google.protobuf.message#Message.MergeFrom.details)
      doc. Note that you cannot override primitive fields with default value,
      such as 0 or empty string, etc., because we use proto3, which doesn't
      support field presence yet.
  """
    analysisQuery = _messages.MessageField('IamPolicyAnalysisQuery', 1)
    outputConfig = _messages.MessageField('IamPolicyAnalysisOutputConfig', 2)
    savedAnalysisQuery = _messages.StringField(3)