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
def FromServiceInfo(service, source_dir, use_flex_with_buildpacks=False):
    """Constructs a BuilderReference from a ServiceYamlInfo.

  Args:
    service: ServiceYamlInfo, The parsed service config.
    source_dir: str, the source containing the application directory to build.
    use_flex_with_buildpacks: bool, if true, use the build-image and
      run-image built through buildpacks.

  Returns:
    RuntimeBuilderVersion for the service.
  """
    runtime_config = service.parsed.runtime_config
    legacy_version = runtime_config.get('runtime_version', None) if runtime_config else None
    resolver = Resolver(service.runtime, source_dir, legacy_version, use_flex_with_buildpacks)
    return resolver.GetBuilderReference()