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
def _GetMSBuildGlobalProperties(spec, version, guid, gyp_file_name):
    namespace = os.path.splitext(gyp_file_name)[0]
    properties = [['PropertyGroup', {'Label': 'Globals'}, ['ProjectGuid', guid], ['Keyword', 'Win32Proj'], ['RootNamespace', namespace], ['IgnoreWarnCompileDuplicatedFilename', 'true']]]
    if os.environ.get('PROCESSOR_ARCHITECTURE') == 'AMD64' or os.environ.get('PROCESSOR_ARCHITEW6432') == 'AMD64':
        properties[0].append(['PreferredToolArchitecture', 'x64'])
    if spec.get('msvs_target_platform_version'):
        target_platform_version = spec.get('msvs_target_platform_version')
        properties[0].append(['WindowsTargetPlatformVersion', target_platform_version])
        if spec.get('msvs_target_platform_minversion'):
            target_platform_minversion = spec.get('msvs_target_platform_minversion')
            properties[0].append(['WindowsTargetPlatformMinVersion', target_platform_minversion])
        else:
            properties[0].append(['WindowsTargetPlatformMinVersion', target_platform_version])
    if spec.get('msvs_enable_winrt'):
        properties[0].append(['DefaultLanguage', 'en-US'])
        properties[0].append(['AppContainerApplication', 'true'])
        if spec.get('msvs_application_type_revision'):
            app_type_revision = spec.get('msvs_application_type_revision')
            properties[0].append(['ApplicationTypeRevision', app_type_revision])
        else:
            properties[0].append(['ApplicationTypeRevision', '8.1'])
        if spec.get('msvs_enable_winphone'):
            properties[0].append(['ApplicationType', 'Windows Phone'])
        else:
            properties[0].append(['ApplicationType', 'Windows Store'])
    platform_name = None
    msvs_windows_sdk_version = None
    for configuration in spec['configurations'].values():
        platform_name = platform_name or _ConfigPlatform(configuration)
        msvs_windows_sdk_version = msvs_windows_sdk_version or _ConfigWindowsTargetPlatformVersion(configuration, version)
        if platform_name and msvs_windows_sdk_version:
            break
    if msvs_windows_sdk_version:
        properties[0].append(['WindowsTargetPlatformVersion', str(msvs_windows_sdk_version)])
    elif version.compatible_sdks:
        raise GypError('%s requires any SDK of %s version, but none were found' % (version.description, version.compatible_sdks))
    if platform_name == 'ARM':
        properties[0].append(['WindowsSDKDesktopARMSupport', 'true'])
    return properties