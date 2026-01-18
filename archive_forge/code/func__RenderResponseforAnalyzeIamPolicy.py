from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def _RenderResponseforAnalyzeIamPolicy(response, analyze_service_account_impersonation, api_version=DEFAULT_API_VERSION):
    """Renders the response of the AnalyzeIamPolicy request."""
    if response.fullyExplored:
        msg = 'Your analysis request is fully explored. '
    else:
        msg = 'Your analysis request is NOT fully explored. You can use the --show-response option to see the unexplored part. '
    has_results = False
    if response.mainAnalysis.analysisResults:
        has_results = True
    if not has_results and analyze_service_account_impersonation:
        for sa_impersonation_analysis in response.serviceAccountImpersonationAnalysis:
            if sa_impersonation_analysis.analysisResults:
                has_results = True
                break
    if not has_results:
        msg += 'No matching ACL is found.'
    else:
        msg += 'The ACLs matching your requests are listed per IAM policy binding, so there could be duplications.'
    for entry in _RenderAnalysisforAnalyzeIamPolicy(response.mainAnalysis, api_version):
        yield entry
    if analyze_service_account_impersonation:
        for analysis in response.serviceAccountImpersonationAnalysis:
            title = {'Service Account Impersonation Analysis Query': analysis.analysisQuery}
            yield title
            for entry in _RenderAnalysisforAnalyzeIamPolicy(analysis, api_version):
                yield entry
    log.status.Print(msg)