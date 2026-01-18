import ntpath
import os
import posixpath
import re
import subprocess
import sys
from collections import OrderedDict
import gyp.common
import gyp.easy_xml as easy_xml
import gyp.generator.ninja as ninja_generator
import gyp.MSVSNew as MSVSNew
import gyp.MSVSProject as MSVSProject
import gyp.MSVSSettings as MSVSSettings
import gyp.MSVSToolFile as MSVSToolFile
import gyp.MSVSUserFile as MSVSUserFile
import gyp.MSVSUtil as MSVSUtil
import gyp.MSVSVersion as MSVSVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def _EscapeEnvironmentVariableExpansion(s):
    """Escapes % characters.

  Escapes any % characters so that Windows-style environment variable
  expansions will leave them alone.
  See http://connect.microsoft.com/VisualStudio/feedback/details/106127/cl-d-name-text-containing-percentage-characters-doesnt-compile
  to understand why we have to do this.

  Args:
      s: The string to be escaped.

  Returns:
      The escaped string.
  """
    s = s.replace('%', '%%')
    return s