from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.run import traffic
import six
def _SplitManagedLatestStatusTarget(spec_dict, status_dict, is_platform_managed, latest_ready_revision_name):
    """Splits the fully-managed latest status target.

  For managed the status target for the latest revision is
  included by revisionName only and may hold the combined traffic
  percent for both latestRevisionName and latestRevision spec targets.
  Here we adjust keys in status_dict to match with spec_dict.

  Args:
    spec_dict: Dictionary mapping revision name or 'LATEST' to the spec
      traffic target referencing that revision.
    status_dict: Dictionary mapping revision name or 'LATEST' to the status
      traffic target referencing that revision. Modified by this function.
    is_platform_managed: Boolean indicating if the current platform is Cloud Run
      fully-managed.
    latest_ready_revision_name: The name of the latest ready revision.

  Returns:
    Optionally, the id of the list of status targets containing the combined
    traffic referencing the latest ready revision by name and by latest.
  """
    combined_status_targets_id = None
    if is_platform_managed and traffic.LATEST_REVISION_KEY in spec_dict and (traffic.LATEST_REVISION_KEY not in status_dict) and (latest_ready_revision_name in status_dict):
        latest_status_targets = status_dict[latest_ready_revision_name]
        status_dict[traffic.LATEST_REVISION_KEY] = latest_status_targets
        if latest_ready_revision_name in spec_dict:
            combined_status_targets_id = id(latest_status_targets)
        else:
            del status_dict[latest_ready_revision_name]
    return combined_status_targets_id