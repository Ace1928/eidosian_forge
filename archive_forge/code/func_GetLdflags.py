import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetLdflags(self, configname, product_dir, gyp_to_build_path, arch=None):
    """Returns flags that need to be passed to the linker.

    Args:
        configname: The name of the configuration to get ld flags for.
        product_dir: The directory where products such static and dynamic
            libraries are placed. This is added to the library search path.
        gyp_to_build_path: A function that converts paths relative to the
            current gyp file to paths relative to the build directory.
    """
    self.configname = configname
    ldflags = []
    for ldflag in self._Settings().get('OTHER_LDFLAGS', []):
        ldflags.append(self._MapLinkerFlagFilename(ldflag, gyp_to_build_path))
    if self._Test('DEAD_CODE_STRIPPING', 'YES', default='NO'):
        ldflags.append('-Wl,-dead_strip')
    if self._Test('PREBINDING', 'YES', default='NO'):
        ldflags.append('-Wl,-prebind')
    self._Appendf(ldflags, 'DYLIB_COMPATIBILITY_VERSION', '-compatibility_version %s')
    self._Appendf(ldflags, 'DYLIB_CURRENT_VERSION', '-current_version %s')
    self._AppendPlatformVersionMinFlags(ldflags)
    if 'SDKROOT' in self._Settings() and self._SdkPath():
        ldflags.append('-isysroot ' + self._SdkPath())
    for library_path in self._Settings().get('LIBRARY_SEARCH_PATHS', []):
        ldflags.append('-L' + gyp_to_build_path(library_path))
    if 'ORDER_FILE' in self._Settings():
        ldflags.append('-Wl,-order_file ' + '-Wl,' + gyp_to_build_path(self._Settings()['ORDER_FILE']))
    if not gyp.common.CrossCompileRequested():
        if arch is not None:
            archs = [arch]
        else:
            assert self.configname
            archs = self.GetActiveArchs(self.configname)
        if len(archs) != 1:
            self._WarnUnimplemented('ARCHS')
            archs = ['i386']
        ldflags.append('-arch ' + archs[0])
    ldflags.append('-L' + (product_dir if product_dir != '.' else './'))
    install_name = self.GetInstallName()
    if install_name and self.spec['type'] != 'loadable_module':
        ldflags.append('-install_name ' + install_name.replace(' ', '\\ '))
    for rpath in self._Settings().get('LD_RUNPATH_SEARCH_PATHS', []):
        ldflags.append('-Wl,-rpath,' + rpath)
    sdk_root = self._SdkPath()
    if not sdk_root:
        sdk_root = ''
    config = self.spec['configurations'][self.configname]
    framework_dirs = config.get('mac_framework_dirs', [])
    for directory in framework_dirs:
        ldflags.append('-F' + directory.replace('$(SDKROOT)', sdk_root))
    if self._IsXCTest():
        platform_root = self._XcodePlatformPath(configname)
        if sdk_root and platform_root:
            ldflags.append('-F' + platform_root + '/Developer/Library/Frameworks/')
            ldflags.append('-framework XCTest')
    is_extension = self._IsIosAppExtension() or self._IsIosWatchKitExtension()
    if sdk_root and is_extension:
        xcode_version, _ = XcodeVersion()
        if xcode_version < '0900':
            ldflags.append('-lpkstart')
            ldflags.append(sdk_root + '/System/Library/PrivateFrameworks/PlugInKit.framework/PlugInKit')
        else:
            ldflags.append('-e _NSExtensionMain')
        ldflags.append('-fapplication-extension')
    self._Appendf(ldflags, 'CLANG_CXX_LIBRARY', '-stdlib=%s')
    self.configname = None
    return ldflags