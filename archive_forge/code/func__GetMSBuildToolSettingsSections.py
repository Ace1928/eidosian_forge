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
def _GetMSBuildToolSettingsSections(spec, configurations):
    groups = []
    for name, configuration in sorted(configurations.items()):
        msbuild_settings = configuration['finalized_msbuild_settings']
        group = ['ItemDefinitionGroup', {'Condition': _GetConfigurationCondition(name, configuration, spec)}]
        for tool_name, tool_settings in sorted(msbuild_settings.items()):
            if tool_name:
                if tool_settings:
                    tool = [tool_name]
                    for name, value in sorted(tool_settings.items()):
                        formatted_value = _GetValueFormattedForMSBuild(tool_name, name, value)
                        tool.append([name, formatted_value])
                    group.append(tool)
        groups.append(group)
    return groups