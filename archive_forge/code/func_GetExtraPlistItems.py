import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetExtraPlistItems(self, configname=None):
    """Returns a dictionary with extra items to insert into Info.plist."""
    if configname not in XcodeSettings._plist_cache:
        cache = {}
        cache['BuildMachineOSBuild'] = self._BuildMachineOSBuild()
        xcode_version, xcode_build = XcodeVersion()
        cache['DTXcode'] = xcode_version
        cache['DTXcodeBuild'] = xcode_build
        compiler = self.xcode_settings[configname].get('GCC_VERSION')
        if compiler is not None:
            cache['DTCompiler'] = compiler
        sdk_root = self._SdkRoot(configname)
        if not sdk_root:
            sdk_root = self._DefaultSdkRoot()
        sdk_version = self._GetSdkVersionInfoItem(sdk_root, '--show-sdk-version')
        cache['DTSDKName'] = sdk_root + (sdk_version or '')
        if xcode_version >= '0720':
            cache['DTSDKBuild'] = self._GetSdkVersionInfoItem(sdk_root, '--show-sdk-build-version')
        elif xcode_version >= '0430':
            cache['DTSDKBuild'] = sdk_version
        else:
            cache['DTSDKBuild'] = cache['BuildMachineOSBuild']
        if self.isIOS:
            cache['MinimumOSVersion'] = self.xcode_settings[configname].get('IPHONEOS_DEPLOYMENT_TARGET')
            cache['DTPlatformName'] = sdk_root
            cache['DTPlatformVersion'] = sdk_version
            if configname.endswith('iphoneos'):
                cache['CFBundleSupportedPlatforms'] = ['iPhoneOS']
                cache['DTPlatformBuild'] = cache['DTSDKBuild']
            else:
                cache['CFBundleSupportedPlatforms'] = ['iPhoneSimulator']
                cache['DTPlatformBuild'] = ''
        XcodeSettings._plist_cache[configname] = cache
    items = dict(XcodeSettings._plist_cache[configname])
    if self.isIOS:
        items['UIDeviceFamily'] = self._XcodeIOSDeviceFamily(configname)
    return items