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
def _GenerateMSVSProject(project, options, version, generator_flags):
    """Generates a .vcproj file.  It may create .rules and .user files too.

  Arguments:
    project: The project object we will generate the file for.
    options: Global options passed to the generator.
    version: The VisualStudioVersion object.
    generator_flags: dict of generator-specific flags.
  """
    spec = project.spec
    gyp.common.EnsureDirExists(project.path)
    platforms = _GetUniquePlatforms(spec)
    p = MSVSProject.Writer(project.path, version, spec['target_name'], project.guid, platforms)
    project_dir = os.path.split(project.path)[0]
    gyp_path = _NormalizedSource(project.build_file)
    relative_path_of_gyp_file = gyp.common.RelativePath(gyp_path, project_dir)
    config_type = _GetMSVSConfigurationType(spec, project.build_file)
    for config_name, config in spec['configurations'].items():
        _AddConfigurationToMSVSProject(p, spec, config_type, config_name, config)
    gyp_file = os.path.split(project.build_file)[1]
    sources, excluded_sources = _PrepareListOfSources(spec, generator_flags, gyp_file)
    actions_to_add = {}
    _GenerateRulesForMSVS(p, project_dir, options, spec, sources, excluded_sources, actions_to_add)
    list_excluded = generator_flags.get('msvs_list_excluded_files', True)
    sources, excluded_sources, excluded_idl = _AdjustSourcesAndConvertToFilterHierarchy(spec, options, project_dir, sources, excluded_sources, list_excluded, version)
    missing_sources = _VerifySourcesExist(sources, project_dir)
    p.AddFiles(sources)
    _AddToolFilesToMSVS(p, spec)
    _HandlePreCompiledHeaders(p, sources, spec)
    _AddActions(actions_to_add, spec, relative_path_of_gyp_file)
    _AddCopies(actions_to_add, spec)
    _WriteMSVSUserFile(project.path, version, spec)
    excluded_sources = _FilterActionsFromExcluded(excluded_sources, actions_to_add)
    _ExcludeFilesFromBeingBuilt(p, spec, excluded_sources, excluded_idl, list_excluded)
    _AddAccumulatedActionsToMSVS(p, spec, actions_to_add)
    p.WriteIfChanged()
    return missing_sources