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
def _GetMSBuildConfigurationGlobalProperties(spec, configurations, build_file):
    new_paths = []
    cygwin_dirs = spec.get('msvs_cygwin_dirs', ['.'])[0]
    if cygwin_dirs:
        cyg_path = '$(MSBuildProjectDirectory)\\%s\\bin\\' % _FixPath(cygwin_dirs)
        new_paths.append(cyg_path)
        python_path = cyg_path.replace('cygwin\\bin', 'python_26')
        new_paths.append(python_path)
        if new_paths:
            new_paths = '$(ExecutablePath);' + ';'.join(new_paths)
    properties = {}
    for name, configuration in sorted(configurations.items()):
        condition = _GetConfigurationCondition(name, configuration, spec)
        attributes = _GetMSBuildAttributes(spec, configuration, build_file)
        msbuild_settings = configuration['finalized_msbuild_settings']
        _AddConditionalProperty(properties, condition, 'IntDir', attributes['IntermediateDirectory'])
        _AddConditionalProperty(properties, condition, 'OutDir', attributes['OutputDirectory'])
        _AddConditionalProperty(properties, condition, 'TargetName', attributes['TargetName'])
        if 'TargetExt' in attributes:
            _AddConditionalProperty(properties, condition, 'TargetExt', attributes['TargetExt'])
        if attributes.get('TargetPath'):
            _AddConditionalProperty(properties, condition, 'TargetPath', attributes['TargetPath'])
        if attributes.get('TargetExt'):
            _AddConditionalProperty(properties, condition, 'TargetExt', attributes['TargetExt'])
        if new_paths:
            _AddConditionalProperty(properties, condition, 'ExecutablePath', new_paths)
        tool_settings = msbuild_settings.get('', {})
        for name, value in sorted(tool_settings.items()):
            formatted_value = _GetValueFormattedForMSBuild('', name, value)
            _AddConditionalProperty(properties, condition, name, formatted_value)
    return _GetMSBuildPropertyGroup(spec, None, properties)