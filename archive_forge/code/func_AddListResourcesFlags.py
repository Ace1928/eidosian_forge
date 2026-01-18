from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddListResourcesFlags(parser):
    _GetBulkExportParentGroup(parser, project_help='Project ID to list supported resources for.', org_help='Organization ID to list supported resources for.', folder_help='Folder ID to list supported resources for.')