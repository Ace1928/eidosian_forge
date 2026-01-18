from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from six.moves.urllib import parse
import uritemplate
def CreateToolResultsUiUrl(project_id, tool_results_ids):
    """Create the URL for a test's Tool Results UI in the Firebase App Manager.

  Args:
    project_id: string containing the user's GCE project ID.
    tool_results_ids: a ToolResultsIds object holding history & execution IDs.

  Returns:
    A url to the Tool Results UI.
  """
    url_base = properties.VALUES.test.results_base_url.Get()
    if not url_base:
        url_base = 'https://console.firebase.google.com'
    url_end = uritemplate.expand('project/{project}/testlab/histories/{history}/matrices/{execution}', {'project': project_id, 'history': tool_results_ids.history_id, 'execution': tool_results_ids.execution_id})
    return parse.urljoin(url_base, url_end)