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
def _GetValueFormattedForMSBuild(tool_name, name, value):
    if type(value) == list:
        if name in ['AdditionalIncludeDirectories', 'AdditionalLibraryDirectories', 'AdditionalOptions', 'DelayLoadDLLs', 'DisableSpecificWarnings', 'PreprocessorDefinitions']:
            value.append('%%(%s)' % name)
        exceptions = {'ClCompile': ['AdditionalOptions'], 'Link': ['AdditionalOptions'], 'Lib': ['AdditionalOptions']}
        if tool_name in exceptions and name in exceptions[tool_name]:
            char = ' '
        else:
            char = ';'
        formatted_value = char.join([MSVSSettings.ConvertVCMacrosToMSBuild(i) for i in value])
    else:
        formatted_value = MSVSSettings.ConvertVCMacrosToMSBuild(value)
    return formatted_value