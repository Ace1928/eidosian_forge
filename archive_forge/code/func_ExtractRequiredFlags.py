from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def ExtractRequiredFlags(parser):
    parent_group = parser.add_mutually_exclusive_group()
    AddOrganizationFlag(parent_group, 'Organization ID')
    AddFolderFlag(parent_group, 'Folder ID')
    AddProjectFlag(parent_group, 'Project ID')