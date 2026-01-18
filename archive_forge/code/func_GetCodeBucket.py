from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def GetCodeBucket(app, project):
    """Gets a bucket reference for a Cloud Build.

  Args:
    app: App resource for this project
    project: str, The name of the current project.

  Returns:
    storage_util.BucketReference, The bucket to use.
  """
    log.debug('No bucket specified, retrieving default bucket.')
    if not app.codeBucket:
        raise exceptions.DefaultBucketAccessError(project)
    return storage_util.BucketReference.FromUrl(app.codeBucket)