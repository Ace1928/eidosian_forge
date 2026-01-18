from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
def _GetImportObjects(parent_object):
    """Given a file object, gets all child objects it imports.

  Args:
    parent_object: The object in which to look for imports.

  Returns:
    A list of import objects representing the files imported by the parent.

  Raises:
    ConfigError: If we cannont read the file, the yaml is malformed, or
       the import object does not contain a 'path' field.
  """
    globbing_enabled = googlecloudsdk.core.properties.VALUES.deployment_manager.glob_imports.GetBool()
    yaml_imports = _GetYamlImports(parent_object, globbing_enabled=globbing_enabled)
    child_objects = []
    for yaml_import in yaml_imports:
        child_path = parent_object.BuildChildPath(yaml_import[PATH])
        child_objects.append(_BuildFileImportObject(child_path, yaml_import[NAME]))
    return child_objects