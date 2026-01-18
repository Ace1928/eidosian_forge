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
def _FixPath(path, separator='\\'):
    """Convert paths to a form that will make sense in a vcproj file.

  Arguments:
    path: The path to convert, may contain / etc.
  Returns:
    The path with all slashes made into backslashes.
  """
    if fixpath_prefix and path and (not os.path.isabs(path)) and (not path[0] == '$') and (not _IsWindowsAbsPath(path)):
        path = os.path.join(fixpath_prefix, path)
    if separator == '\\':
        path = path.replace('/', '\\')
    path = _NormalizedSource(path)
    if separator == '/':
        path = path.replace('\\', '/')
    if path and path[-1] == separator:
        path = path[:-1]
    return path