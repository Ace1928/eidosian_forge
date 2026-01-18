from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
import sys
import time
from apitools.base.py import list_pager
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import times
import six
def LatestStableTensorflowVersion(self, zone):
    """Parses available Tensorflow versions to find the most stable version."""
    project = properties.VALUES.core.project.Get(required=True)
    parent_ref = resources.REGISTRY.Parse(zone, params={'projectsId': project}, collection='tpu.projects.locations')
    request = self.messages.TpuProjectsLocationsTensorflowVersionsListRequest(parent=parent_ref.RelativeName())
    tf_versions = list_pager.YieldFromList(service=self.client.projects_locations_tensorflowVersions, request=request, batch_size_attribute='pageSize', field='tensorflowVersions')
    parsed_tf_versions = []
    for tf_version in tf_versions:
        parsed_tf_versions.append(TensorflowVersionParser.ParseVersion(tf_version.version))
    sorted_tf_versions = sorted(parsed_tf_versions)
    for version in sorted_tf_versions:
        if version.is_nightly:
            raise HttpNotFoundError('No stable release found', None, None)
        if not version.modifier:
            return version.VersionString()
    raise HttpNotFoundError('No stable release found', None, None)