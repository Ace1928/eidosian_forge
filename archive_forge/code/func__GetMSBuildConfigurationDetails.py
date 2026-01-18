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
def _GetMSBuildConfigurationDetails(spec, build_file):
    properties = {}
    for name, settings in spec['configurations'].items():
        msbuild_attributes = _GetMSBuildAttributes(spec, settings, build_file)
        condition = _GetConfigurationCondition(name, settings, spec)
        character_set = msbuild_attributes.get('CharacterSet')
        config_type = msbuild_attributes.get('ConfigurationType')
        _AddConditionalProperty(properties, condition, 'ConfigurationType', config_type)
        if config_type == 'Driver':
            _AddConditionalProperty(properties, condition, 'DriverType', 'WDM')
            _AddConditionalProperty(properties, condition, 'TargetVersion', _ConfigTargetVersion(settings))
        if character_set:
            if 'msvs_enable_winrt' not in spec:
                _AddConditionalProperty(properties, condition, 'CharacterSet', character_set)
    return _GetMSBuildPropertyGroup(spec, 'Configuration', properties)