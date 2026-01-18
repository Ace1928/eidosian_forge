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
def _AdjustSourcesAndConvertToFilterHierarchy(spec, options, gyp_dir, sources, excluded_sources, list_excluded, version):
    """Adjusts the list of sources and excluded sources.

  Also converts the sets to lists.

  Arguments:
    spec: The target dictionary containing the properties of the target.
    options: Global generator options.
    gyp_dir: The path to the gyp file being processed.
    sources: A set of sources to be included for this project.
    excluded_sources: A set of sources to be excluded for this project.
    version: A MSVSVersion object.
  Returns:
    A trio of (list of sources, list of excluded sources,
               path of excluded IDL file)
  """
    excluded_sources.update(OrderedSet(spec.get('sources_excluded', [])))
    sources.update(excluded_sources)
    sources = _FixPaths(sources)
    excluded_sources = _FixPaths(excluded_sources)
    excluded_idl = _IdlFilesHandledNonNatively(spec, sources)
    precompiled_related = _GetPrecompileRelatedFiles(spec)
    fully_excluded = [i for i in excluded_sources if i not in precompiled_related]
    sources = [i.split('\\') for i in sources]
    sources = _ConvertSourcesToFilterHierarchy(sources, excluded=fully_excluded, list_excluded=list_excluded, msvs_version=version)
    if version.UsesVcxproj():
        while all([isinstance(s, MSVSProject.Filter) for s in sources]) and len({s.name for s in sources}) == 1:
            assert all([len(s.contents) == 1 for s in sources])
            sources = [s.contents[0] for s in sources]
    else:
        while len(sources) == 1 and isinstance(sources[0], MSVSProject.Filter):
            sources = sources[0].contents
    return (sources, excluded_sources, excluded_idl)