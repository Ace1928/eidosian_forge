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
def _EscapeVCProjCommandLineArgListItem(s):
    """Escapes command line arguments for MSVS.

  The VCProj format stores string lists in a single string using commas and
  semi-colons as separators, which must be quoted if they are to be
  interpreted literally. However, command-line arguments may already have
  quotes, and the VCProj parser is ignorant of the backslash escaping
  convention used by CommandLineToArgv, so the command-line quotes and the
  VCProj quotes may not be the same quotes. So to store a general
  command-line argument in a VCProj list, we need to parse the existing
  quoting according to VCProj's convention and quote any delimiters that are
  not already quoted by that convention. The quotes that we add will also be
  seen by CommandLineToArgv, so if backslashes precede them then we also have
  to escape those backslashes according to the CommandLineToArgv
  convention.

  Args:
      s: the string to be escaped.
  Returns:
      the escaped string.
  """

    def _Replace(match):
        return 2 * match.group(1) + '"' + match.group(2) + '"'
    segments = s.split('"')
    for i in range(0, len(segments), 2):
        segments[i] = delimiters_replacer_regex.sub(_Replace, segments[i])
    s = '"'.join(segments)
    if len(segments) % 2 == 0:
        print('Warning: MSVS may misinterpret the odd number of ' + 'quotes in ' + s, file=sys.stderr)
    return s