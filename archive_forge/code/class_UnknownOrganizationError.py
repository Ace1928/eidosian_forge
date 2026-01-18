from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
class UnknownOrganizationError(exceptions.BadArgumentException):

    def __init__(self, org_argument, metavar='ORGANIZATION_ID'):
        message = 'Cannot determine Organization ID from [{0}]. Try `gcloud organizations list` to find your Organization ID.'.format(org_argument)
        super(UnknownOrganizationError, self).__init__(metavar, message)