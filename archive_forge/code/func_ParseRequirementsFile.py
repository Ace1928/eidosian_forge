from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def ParseRequirementsFile(requirements_file_path):
    """Parses the given requirements file into a requirements dictionary.

  If the file path is GCS file path, use GCS file parser to parse requirements
  file. Otherwise, use local file parser.

  Args:
    requirements_file_path: Filepath to the requirements file.

  Returns:
    {string: string}, dict mapping from PyPI package name to extras and version
    specifier, if provided.

  Raises:
    Error: if requirements file cannot be read.
  """
    try:
        is_gcs_file_path = requirements_file_path.startswith('gs://')
        if is_gcs_file_path:
            storage_client = storage_api.StorageClient()
            object_ref = storage_util.ObjectReference.FromUrl(requirements_file_path)
            file_content = storage_client.ReadObject(object_ref)
        else:
            file_content = files.FileReader(requirements_file_path)
        requirements = {}
        with file_content as requirements_file:
            for requirement_specifier in requirements_file:
                requirement_specifier = requirement_specifier.strip()
                if not requirement_specifier or requirement_specifier.startswith('#'):
                    continue
                package, version = SplitRequirementSpecifier(requirement_specifier)
                if package in requirements:
                    raise Error('Duplicate package in requirements file: {0}'.format(package))
                requirements[package] = version
            return requirements
    except (files.Error, storage_api.Error, storage_util.Error):
        core_exceptions.reraise(Error('Unable to read requirements file {0}'.format(requirements_file_path)))