from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def GetPathMatchers():
    """Get list of path matchers ordered by descending precedence.

  Returns:
    List[Function], ordered list of functions on the form fn(path, stager),
    where fn returns a Service or None if no match.
  """
    return [ServiceYamlMatcher, AppengineWebMatcher, JarMatcher, PomXmlMatcher, BuildGradleMatcher, ExplicitAppYamlMatcher, UnidentifiedDirMatcher]