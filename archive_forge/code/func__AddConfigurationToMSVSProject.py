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
def _AddConfigurationToMSVSProject(p, spec, config_type, config_name, config):
    """Adds a configuration to the MSVS project.

  Many settings in a vcproj file are specific to a configuration.  This
  function the main part of the vcproj file that's configuration specific.

  Arguments:
    p: The target project being generated.
    spec: The target dictionary containing the properties of the target.
    config_type: The configuration type, a number as defined by Microsoft.
    config_name: The name of the configuration.
    config: The dictionary that defines the special processing to be done
            for this configuration.
  """
    include_dirs, midl_include_dirs, resource_include_dirs = _GetIncludeDirs(config)
    libraries = _GetLibraries(spec)
    library_dirs = _GetLibraryDirs(config)
    out_file, vc_tool, _ = _GetOutputFilePathAndTool(spec, msbuild=False)
    defines = _GetDefines(config)
    defines = [_EscapeCppDefineForMSVS(d) for d in defines]
    disabled_warnings = _GetDisabledWarnings(config)
    prebuild = config.get('msvs_prebuild')
    postbuild = config.get('msvs_postbuild')
    def_file = _GetModuleDefinition(spec)
    precompiled_header = config.get('msvs_precompiled_header')
    tools = dict()
    msvs_settings = config.get('msvs_settings', {})
    MSVSSettings.ValidateMSVSSettings(msvs_settings)
    _ToolAppend(tools, 'VCLinkerTool', 'AdditionalDependencies', ['$(NOINHERIT)'])
    for tool in msvs_settings:
        settings = config['msvs_settings'][tool]
        for setting in settings:
            _ToolAppend(tools, tool, setting, settings[setting])
    _ToolAppend(tools, 'VCCLCompilerTool', 'AdditionalIncludeDirectories', include_dirs)
    _ToolAppend(tools, 'VCMIDLTool', 'AdditionalIncludeDirectories', midl_include_dirs)
    _ToolAppend(tools, 'VCResourceCompilerTool', 'AdditionalIncludeDirectories', resource_include_dirs)
    _ToolAppend(tools, 'VCLinkerTool', 'AdditionalDependencies', libraries)
    _ToolAppend(tools, 'VCLinkerTool', 'AdditionalLibraryDirectories', library_dirs)
    if out_file:
        _ToolAppend(tools, vc_tool, 'OutputFile', out_file, only_if_unset=True)
    _ToolAppend(tools, 'VCCLCompilerTool', 'PreprocessorDefinitions', defines)
    _ToolAppend(tools, 'VCResourceCompilerTool', 'PreprocessorDefinitions', defines)
    _ToolAppend(tools, 'VCCLCompilerTool', 'ProgramDataBaseFileName', '$(IntDir)$(ProjectName)\\vc80.pdb', only_if_unset=True)
    _ToolAppend(tools, 'VCCLCompilerTool', 'DisableSpecificWarnings', disabled_warnings)
    _ToolAppend(tools, 'VCPreBuildEventTool', 'CommandLine', prebuild)
    _ToolAppend(tools, 'VCPostBuildEventTool', 'CommandLine', postbuild)
    if precompiled_header:
        precompiled_header = os.path.split(precompiled_header)[1]
        _ToolAppend(tools, 'VCCLCompilerTool', 'UsePrecompiledHeader', '2')
        _ToolAppend(tools, 'VCCLCompilerTool', 'PrecompiledHeaderThrough', precompiled_header)
        _ToolAppend(tools, 'VCCLCompilerTool', 'ForcedIncludeFiles', precompiled_header)
    if spec['type'] == 'loadable_module':
        _ToolAppend(tools, 'VCLinkerTool', 'IgnoreImportLibrary', 'true')
    if def_file:
        _ToolAppend(tools, 'VCLinkerTool', 'ModuleDefinitionFile', def_file)
    _AddConfigurationToMSVS(p, spec, tools, config, config_type, config_name)