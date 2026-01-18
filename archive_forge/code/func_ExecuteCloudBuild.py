from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import time
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import range  # pylint: disable=redefined-builtin
def ExecuteCloudBuild(self, build, project=None):
    """Execute a call to CloudBuild service and wait for it to finish.


    Args:
      build: Build object. The Build to execute.
      project: The project to execute, or None to use the current project
          property.

    Raises:
      BuildFailedError: when the build fails.
    """
    build_op = self.ExecuteCloudBuildAsync(build, project)
    self.WaitAndStreamLogs(build_op)