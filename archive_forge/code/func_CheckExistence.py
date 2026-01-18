from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
from googlecloudsdk.command_lib.projects import util
def CheckExistence(instance):
    """Checks whether a service account exists on the instance."""
    response_message = '> Is a service account present on the instance? '
    if not instance.serviceAccounts:
        response_message += 'No\nNo service account is present on the instance. Visit https://cloud.google.com/compute/docs/access/create-enable-service-accounts-for-instances on how to create a service account for an instance.'
        return utils.Response(False, response_message)
    response_message += 'Yes'
    return utils.Response(True, response_message)