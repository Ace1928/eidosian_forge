from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import build as app_build
from googlecloudsdk.api_lib.app import cloud_build
from googlecloudsdk.api_lib.app import docker_image
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.api_lib.app.runtimes import fingerprinter
from googlecloudsdk.api_lib.cloudbuild import build as cloudbuild_build
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as s_exceptions
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.tools import context_util
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
def _GetYamlPath(source_dir, service_path, skip_files, gen_files):
    """Returns the yaml path, optionally updating gen_files.

  Args:
    source_dir: str, the absolute path to the root of the application directory.
    service_path: str, the absolute path to the service YAML file
    skip_files: appengine.api.Validation._RegexStr, the validated regex object
      from the service info file.
    gen_files: dict, the dict of files to generate. May be updated if a file
      needs to be generated.

  Returns:
    str, the relative path to the service YAML file that should be used for
      build.
  """
    if files.IsDirAncestorOf(source_dir, service_path):
        rel_path = os.path.relpath(service_path, start=source_dir)
        if not util.ShouldSkip(skip_files, rel_path):
            return rel_path
    yaml_contents = files.ReadFileContents(service_path)
    checksum = files.Checksum().AddContents(yaml_contents.encode()).HexDigest()
    generated_path = '_app_{}.yaml'.format(checksum)
    gen_files[generated_path] = yaml_contents
    return generated_path