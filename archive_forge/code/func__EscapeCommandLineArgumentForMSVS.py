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
def _EscapeCommandLineArgumentForMSVS(s):
    """Escapes a Windows command-line argument.

  So that the Win32 CommandLineToArgv function will turn the escaped result back
  into the original string.
  See http://msdn.microsoft.com/en-us/library/17w5ykft.aspx
  ("Parsing C++ Command-Line Arguments") to understand why we have to do
  this.

  Args:
      s: the string to be escaped.
  Returns:
      the escaped string.
  """

    def _Replace(match):
        return 2 * match.group(1) + '\\"'
    s = quote_replacer_regex.sub(_Replace, s)
    s = '"' + s + '"'
    return s