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
def _VerifySourcesExist(sources, root_dir):
    """Verifies that all source files exist on disk.

  Checks that all regular source files, i.e. not created at run time,
  exist on disk.  Missing files cause needless recompilation but no otherwise
  visible errors.

  Arguments:
    sources: A recursive list of Filter/file names.
    root_dir: The root directory for the relative path names.
  Returns:
    A list of source files that cannot be found on disk.
  """
    missing_sources = []
    for source in sources:
        if isinstance(source, MSVSProject.Filter):
            missing_sources.extend(_VerifySourcesExist(source.contents, root_dir))
        elif '$' not in source:
            full_path = os.path.join(root_dir, source)
            if not os.path.exists(full_path):
                missing_sources.append(full_path)
    return missing_sources