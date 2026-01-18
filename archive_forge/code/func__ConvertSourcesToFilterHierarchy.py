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
def _ConvertSourcesToFilterHierarchy(sources, prefix=None, excluded=None, list_excluded=True, msvs_version=None):
    """Converts a list split source file paths into a vcproj folder hierarchy.

  Arguments:
    sources: A list of source file paths split.
    prefix: A list of source file path layers meant to apply to each of sources.
    excluded: A set of excluded files.
    msvs_version: A MSVSVersion object.

  Returns:
    A hierarchy of filenames and MSVSProject.Filter objects that matches the
    layout of the source tree.
    For example:
    _ConvertSourcesToFilterHierarchy([['a', 'bob1.c'], ['b', 'bob2.c']],
                                     prefix=['joe'])
    -->
    [MSVSProject.Filter('a', contents=['joe\\a\\bob1.c']),
     MSVSProject.Filter('b', contents=['joe\\b\\bob2.c'])]
  """
    if not prefix:
        prefix = []
    result = []
    excluded_result = []
    folders = OrderedDict()
    for s in sources:
        if len(s) == 1:
            filename = _NormalizedSource('\\'.join(prefix + s))
            if filename in excluded:
                excluded_result.append(filename)
            else:
                result.append(filename)
        elif msvs_version and (not msvs_version.UsesVcxproj()):
            if not folders.get(s[0]):
                folders[s[0]] = []
            folders[s[0]].append(s[1:])
        else:
            contents = _ConvertSourcesToFilterHierarchy([s[1:]], prefix + [s[0]], excluded=excluded, list_excluded=list_excluded, msvs_version=msvs_version)
            contents = MSVSProject.Filter(s[0], contents=contents)
            result.append(contents)
    if excluded_result and list_excluded:
        excluded_folder = MSVSProject.Filter('_excluded_files', contents=excluded_result)
        result.append(excluded_folder)
    if msvs_version and msvs_version.UsesVcxproj():
        return result
    for f in folders:
        contents = _ConvertSourcesToFilterHierarchy(folders[f], prefix=prefix + [f], excluded=excluded, list_excluded=list_excluded, msvs_version=msvs_version)
        contents = MSVSProject.Filter(f, contents=contents)
        result.append(contents)
    return result