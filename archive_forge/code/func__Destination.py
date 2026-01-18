from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.eventarc import triggers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.command_lib.eventarc import types
def _Destination(trigger):
    """Generate a destination string for the trigger.

  Based on different destination types, this function returns a destination
  string accordingly:

    * Cloud Run trigger: "Cloud Run: {cloud run service or job}"
    * GKE trigger: "GKE: {gke service}"
    * Workflows trigger: "Workflows: {workflow name}"
    * Cloud Functions trigger: "Cloud Functions: {cloud function name}"

  For unknown destination (e.g. new types of destination and corrupted
  destination), this function returns an empty string.

  Args:
    trigger: eventarc trigger proto in python map format.

  Returns:
    A string representing the destination for the trigger.
  """
    destination = trigger.get('destination')
    if destination is None:
        return ''
    if 'cloudRun' in destination:
        dest = destination.get('cloudRun')
        job = dest.get('job')
        if job:
            return 'Cloud Run job: {}'.format(job)
        service = dest.get('service')
        return 'Cloud Run service: {}'.format(service)
    elif 'gke' in destination:
        dest = destination.get('gke')
        return 'GKE: {}'.format(dest.get('service'))
    elif 'cloudFunction' in destination:
        cloud_function_str_pattern = '^projects/.*/locations/.*/functions/(.*)$'
        dest = destination.get('cloudFunction')
        match = re.search(cloud_function_str_pattern, dest)
        return 'Cloud Functions: {}'.format(match.group(1)) if match else ''
    elif 'workflow' in destination:
        workflows_str_pattern = '^projects/.*/locations/.*/workflows/(.*)$'
        dest = destination.get('workflow')
        match = re.search(workflows_str_pattern, dest)
        return 'Workflows: {}'.format(match.group(1)) if match else ''
    elif 'httpEndpoint' in destination:
        dest = destination.get('httpEndpoint')
        return 'HTTP endpoint: {}'.format(dest.get('uri'))
    else:
        return ''