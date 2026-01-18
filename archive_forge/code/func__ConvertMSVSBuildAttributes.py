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
def _ConvertMSVSBuildAttributes(spec, config, build_file):
    config_type = _GetMSVSConfigurationType(spec, build_file)
    msvs_attributes = _GetMSVSAttributes(spec, config, config_type)
    msbuild_attributes = {}
    for a in msvs_attributes:
        if a in ['IntermediateDirectory', 'OutputDirectory']:
            directory = MSVSSettings.ConvertVCMacrosToMSBuild(msvs_attributes[a])
            if not directory.endswith('\\'):
                directory += '\\'
            msbuild_attributes[a] = directory
        elif a == 'CharacterSet':
            msbuild_attributes[a] = _ConvertMSVSCharacterSet(msvs_attributes[a])
        elif a == 'ConfigurationType':
            msbuild_attributes[a] = _ConvertMSVSConfigurationType(msvs_attributes[a])
        else:
            print('Warning: Do not know how to convert MSVS attribute ' + a)
    return msbuild_attributes