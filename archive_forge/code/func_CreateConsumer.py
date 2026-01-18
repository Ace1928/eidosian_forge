from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
def CreateConsumer(project, folder, organization):
    if project:
        _ValidateProject(project)
        return 'projects/' + project
    if folder:
        _ValidateFolder(folder)
        return 'folders/' + folder
    if organization:
        _ValidateOrganization(organization)
        return 'organizations/' + organization
    return None