from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def _GetReferenceFromLegacyWithVersion(self, version):
    """Gets the name of configuration file to use for legacy mode.

    Args:
      version: str, The pinned version of the configuration file.

    Returns:
      BuilderReference
    """
    file_name = '-'.join([self.runtime, version]) + '.yaml'
    file_uri = _Join(self.build_file_root, file_name)
    log.debug('Calculated builder definition using legacy version [%s]', file_uri)
    return BuilderReference(self.runtime, file_uri)