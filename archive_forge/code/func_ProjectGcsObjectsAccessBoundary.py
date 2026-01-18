from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def ProjectGcsObjectsAccessBoundary(project):
    """Get an access boundary limited to to a project's GCS objects.

  Args:
    project: The project ID for the access boundary.

  Returns:
    A JSON formatted access boundary suitable for creating a downscoped token.
  """
    cab_resource = '//cloudresourcemanager.googleapis.com/projects/{}'.format(project)
    access_boundary = {'access_boundary': {'accessBoundaryRules': [{'availableResource': cab_resource, 'availablePermissions': ['inRole:roles/storage.objectViewer', 'inRole:roles/storage.objectCreator', 'inRole:roles/storage.objectAdmin', 'inRole:roles/storage.legacyBucketReader']}]}}
    return six.text_type(json.dumps(access_boundary))