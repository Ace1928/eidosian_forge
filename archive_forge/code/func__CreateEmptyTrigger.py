from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
def _CreateEmptyTrigger(self):
    """Attempt to create an empty trigger in us-central1 to kick off P4SA JIT provision.

    The create request will always fail due to the empty trigger message
    payload, but it will trigger the P4SA JIT provision.

    Returns:
      A long-running operation for create.
    """
    project = properties.VALUES.core.project.Get(required=True)
    parent = 'projects/{}/locations/{}'.format(project, _LOCATION)
    req = self._messages.EventarcProjectsLocationsTriggersCreateRequest(parent=parent, triggerId=_TRIGGER_ID)
    return self._service.Create(req)