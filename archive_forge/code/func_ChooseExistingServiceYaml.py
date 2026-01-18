from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import glob
import os
from googlecloudsdk.core import exceptions
def ChooseExistingServiceYaml(context, arg):
    """Rules for choosing a service.yaml or app.yaml file.

  The rules are meant to discover common filename variants like
  'service.dev.yml' or 'staging-service.yaml'.

  Args:
    context: Build context dir. Could be '.'.
    arg: User's path (relative to context or absolute) to a yaml file with
      service config, or None. The service config could be a knative Service
      description or an appengine app.yaml.

  Returns:
    Absolute path to a yaml file, or None.
  """
    if arg is not None:
        complete_abs_path = os.path.abspath(os.path.join(context, arg))
        if os.path.exists(complete_abs_path):
            return complete_abs_path
        raise exceptions.Error("File '{}' not found.".format(complete_abs_path))
    for pattern in ['*service.dev.yaml', '*service.dev.yml', '*service.yaml', '*service.yml']:
        matches = glob.glob(os.path.join(context, pattern))
        if matches:
            return sorted(matches)[0]
    return None