from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import datetime
import io
import os.path
import shutil
import tarfile
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import snapshot
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import staging_bucket_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _GetRunTargetProperties(targets, project, location):
    """Gets target properties for targets."""
    target_to_target_properties = {}
    for target_id in targets:
        target_ref = target_util.TargetReference(target_id, project, location)
        target = target_util.GetTarget(target_ref)
        target_location = getattr(target, 'run', None)
        if not target_location:
            raise core_exceptions.Error('Target is not of type {}'.format('run'))
        location_attr = getattr(target_location, 'location', None)
        if not location_attr:
            raise core_exceptions.Error('Target location {} does not have a location attribute.'.format(target_location))
        target_to_target_properties[target_id] = _TargetProperties(target_id, location_attr)
    return target_to_target_properties