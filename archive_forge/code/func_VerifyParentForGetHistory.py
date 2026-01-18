from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import properties
def VerifyParentForGetHistory(organization, project, attribute='root cloud asset'):
    """Verify the parent name."""
    if organization is None and project is None:
        raise gcloud_exceptions.RequiredArgumentException('--organization or --project', 'Should specify the organization, or project for {0}.'.format(attribute))
    if organization and project:
        raise gcloud_exceptions.ConflictingArgumentsException('organization', 'project')