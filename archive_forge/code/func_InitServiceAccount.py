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
def InitServiceAccount(self):
    """Force create the Eventarc P4SA, and grant IAM roles to it.

    1) First, trigger the P4SA JIT provision by trying to create an empty
    trigger, ignore the HttpBadRequestError exception, then call
    GenerateServiceIdentity to verify that P4SA creation is completed.
    2) Then grant necessary roles needed to the P4SA for creating GKE triggers.

    Raises:
      GKEDestinationInitializationError: P4SA failed to be created.
    """
    try:
        self._CreateEmptyTrigger()
    except apitools_exceptions.HttpBadRequestError:
        pass
    service_name = common.GetApiServiceName(self._api_version)
    p4sa_email = _GetOrCreateP4SA(service_name)
    if not p4sa_email:
        raise GKEDestinationInitializationError('Failed to initialize project for Cloud Run for Anthos/GKE destinations.')
    self._BindRolesToServiceAccount(p4sa_email, _ROLES)