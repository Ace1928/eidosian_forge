from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_app_update_api_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
def PatchApplication(release_track, split_health_checks=None, service_account=None):
    """Updates an App Engine application via API client.

  Args:
    release_track: The release track of the app update command to run.
    split_health_checks: Boolean, whether to enable split health checks by
      default.
    service_account: str, the app-level default service account to update for
      this App Engine app.
  """
    api_client = appengine_app_update_api_client.GetApiClientForTrack(release_track)
    if split_health_checks is not None or service_account is not None:
        with progress_tracker.ProgressTracker('Updating the app [{0}]'.format(api_client.project)):
            api_client.PatchApplication(split_health_checks=split_health_checks, service_account=service_account)
    else:
        log.status.Print('Nothing to update.')