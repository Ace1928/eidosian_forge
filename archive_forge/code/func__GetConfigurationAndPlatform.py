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
def _GetConfigurationAndPlatform(name, settings, spec):
    configuration = name.rsplit('_', 1)[0]
    platform = settings.get('msvs_configuration_platform', 'Win32')
    if spec['toolset'] == 'host' and platform == 'arm64':
        platform = 'x64'
    return (configuration, platform)