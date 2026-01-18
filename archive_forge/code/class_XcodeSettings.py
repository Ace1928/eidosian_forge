import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
class XcodeSettings:
    """A class that understands the gyp 'xcode_settings' object."""
    _sdk_path_cache = {}
    _platform_path_cache = {}
    _sdk_root_cache = {}
    _plist_cache = {}
    _codesigning_key_cache = {}

    def __init__(self, spec):
        self.spec = spec
        self.isIOS = False
        self.mac_toolchain_dir = None
        self.header_map_path = None
        self.xcode_settings = {}
        configs = spec['configurations']
        for configname, config in configs.items():
            self.xcode_settings[configname] = config.get('xcode_settings', {})
            self._ConvertConditionalKeys(configname)
            if self.xcode_settings[configname].get('IPHONEOS_DEPLOYMENT_TARGET', None):
                self.isIOS = True
        self.configname = None
        self.library_re = re.compile('^lib([^/]+)\\.(a|dylib)$')

    def _ConvertConditionalKeys(self, configname):
        """Converts or warns on conditional keys.  Xcode supports conditional keys,
    such as CODE_SIGN_IDENTITY[sdk=iphoneos*].  This is a partial implementation
    with some keys converted while the rest force a warning."""
        settings = self.xcode_settings[configname]
        conditional_keys = [key for key in settings if key.endswith(']')]
        for key in conditional_keys:
            if key.endswith('[sdk=iphoneos*]'):
                if configname.endswith('iphoneos'):
                    new_key = key.split('[')[0]
                    settings[new_key] = settings[key]
            else:
                print('Warning: Conditional keys not implemented, ignoring:', ' '.join(conditional_keys))
            del settings[key]

    def _Settings(self):
        assert self.configname
        return self.xcode_settings[self.configname]

    def _Test(self, test_key, cond_key, default):
        return self._Settings().get(test_key, default) == cond_key

    def _Appendf(self, lst, test_key, format_str, default=None):
        if test_key in self._Settings():
            lst.append(format_str % str(self._Settings()[test_key]))
        elif default:
            lst.append(format_str % str(default))

    def _WarnUnimplemented(self, test_key):
        if test_key in self._Settings():
            print('Warning: Ignoring not yet implemented key "%s".' % test_key)

    def IsBinaryOutputFormat(self, configname):
        default = 'binary' if self.isIOS else 'xml'
        format = self.xcode_settings[configname].get('INFOPLIST_OUTPUT_FORMAT', default)
        return format == 'binary'

    def IsIosFramework(self):
        return self.spec['type'] == 'shared_library' and self._IsBundle() and self.isIOS

    def _IsBundle(self):
        return int(self.spec.get('mac_bundle', 0)) != 0 or self._IsXCTest() or self._IsXCUiTest()

    def _IsXCTest(self):
        return int(self.spec.get('mac_xctest_bundle', 0)) != 0

    def _IsXCUiTest(self):
        return int(self.spec.get('mac_xcuitest_bundle', 0)) != 0

    def _IsIosAppExtension(self):
        return int(self.spec.get('ios_app_extension', 0)) != 0

    def _IsIosWatchKitExtension(self):
        return int(self.spec.get('ios_watchkit_extension', 0)) != 0

    def _IsIosWatchApp(self):
        return int(self.spec.get('ios_watch_app', 0)) != 0

    def GetFrameworkVersion(self):
        """Returns the framework version of the current target. Only valid for
    bundles."""
        assert self._IsBundle()
        return self.GetPerTargetSetting('FRAMEWORK_VERSION', default='A')

    def GetWrapperExtension(self):
        """Returns the bundle extension (.app, .framework, .plugin, etc).  Only
    valid for bundles."""
        assert self._IsBundle()
        if self.spec['type'] in ('loadable_module', 'shared_library'):
            default_wrapper_extension = {'loadable_module': 'bundle', 'shared_library': 'framework'}[self.spec['type']]
            wrapper_extension = self.GetPerTargetSetting('WRAPPER_EXTENSION', default=default_wrapper_extension)
            return '.' + self.spec.get('product_extension', wrapper_extension)
        elif self.spec['type'] == 'executable':
            if self._IsIosAppExtension() or self._IsIosWatchKitExtension():
                return '.' + self.spec.get('product_extension', 'appex')
            else:
                return '.' + self.spec.get('product_extension', 'app')
        else:
            assert False, "Don't know extension for '{}', target '{}'".format(self.spec['type'], self.spec['target_name'])

    def GetProductName(self):
        """Returns PRODUCT_NAME."""
        return self.spec.get('product_name', self.spec['target_name'])

    def GetFullProductName(self):
        """Returns FULL_PRODUCT_NAME."""
        if self._IsBundle():
            return self.GetWrapperName()
        else:
            return self._GetStandaloneBinaryPath()

    def GetWrapperName(self):
        """Returns the directory name of the bundle represented by this target.
    Only valid for bundles."""
        assert self._IsBundle()
        return self.GetProductName() + self.GetWrapperExtension()

    def GetBundleContentsFolderPath(self):
        """Returns the qualified path to the bundle's contents folder. E.g.
    Chromium.app/Contents or Foo.bundle/Versions/A. Only valid for bundles."""
        if self.isIOS:
            return self.GetWrapperName()
        assert self._IsBundle()
        if self.spec['type'] == 'shared_library':
            return os.path.join(self.GetWrapperName(), 'Versions', self.GetFrameworkVersion())
        else:
            return os.path.join(self.GetWrapperName(), 'Contents')

    def GetBundleResourceFolder(self):
        """Returns the qualified path to the bundle's resource folder. E.g.
    Chromium.app/Contents/Resources. Only valid for bundles."""
        assert self._IsBundle()
        if self.isIOS:
            return self.GetBundleContentsFolderPath()
        return os.path.join(self.GetBundleContentsFolderPath(), 'Resources')

    def GetBundleExecutableFolderPath(self):
        """Returns the qualified path to the bundle's executables folder. E.g.
    Chromium.app/Contents/MacOS. Only valid for bundles."""
        assert self._IsBundle()
        if self.spec['type'] in 'shared_library' or self.isIOS:
            return self.GetBundleContentsFolderPath()
        elif self.spec['type'] in ('executable', 'loadable_module'):
            return os.path.join(self.GetBundleContentsFolderPath(), 'MacOS')

    def GetBundleJavaFolderPath(self):
        """Returns the qualified path to the bundle's Java resource folder.
    E.g. Chromium.app/Contents/Resources/Java. Only valid for bundles."""
        assert self._IsBundle()
        return os.path.join(self.GetBundleResourceFolder(), 'Java')

    def GetBundleFrameworksFolderPath(self):
        """Returns the qualified path to the bundle's frameworks folder. E.g,
    Chromium.app/Contents/Frameworks. Only valid for bundles."""
        assert self._IsBundle()
        return os.path.join(self.GetBundleContentsFolderPath(), 'Frameworks')

    def GetBundleSharedFrameworksFolderPath(self):
        """Returns the qualified path to the bundle's frameworks folder. E.g,
    Chromium.app/Contents/SharedFrameworks. Only valid for bundles."""
        assert self._IsBundle()
        return os.path.join(self.GetBundleContentsFolderPath(), 'SharedFrameworks')

    def GetBundleSharedSupportFolderPath(self):
        """Returns the qualified path to the bundle's shared support folder. E.g,
    Chromium.app/Contents/SharedSupport. Only valid for bundles."""
        assert self._IsBundle()
        if self.spec['type'] == 'shared_library':
            return self.GetBundleResourceFolder()
        else:
            return os.path.join(self.GetBundleContentsFolderPath(), 'SharedSupport')

    def GetBundlePlugInsFolderPath(self):
        """Returns the qualified path to the bundle's plugins folder. E.g,
    Chromium.app/Contents/PlugIns. Only valid for bundles."""
        assert self._IsBundle()
        return os.path.join(self.GetBundleContentsFolderPath(), 'PlugIns')

    def GetBundleXPCServicesFolderPath(self):
        """Returns the qualified path to the bundle's XPC services folder. E.g,
    Chromium.app/Contents/XPCServices. Only valid for bundles."""
        assert self._IsBundle()
        return os.path.join(self.GetBundleContentsFolderPath(), 'XPCServices')

    def GetBundlePlistPath(self):
        """Returns the qualified path to the bundle's plist file. E.g.
    Chromium.app/Contents/Info.plist. Only valid for bundles."""
        assert self._IsBundle()
        if self.spec['type'] in ('executable', 'loadable_module') or self.IsIosFramework():
            return os.path.join(self.GetBundleContentsFolderPath(), 'Info.plist')
        else:
            return os.path.join(self.GetBundleContentsFolderPath(), 'Resources', 'Info.plist')

    def GetProductType(self):
        """Returns the PRODUCT_TYPE of this target."""
        if self._IsIosAppExtension():
            assert self._IsBundle(), 'ios_app_extension flag requires mac_bundle (target %s)' % self.spec['target_name']
            return 'com.apple.product-type.app-extension'
        if self._IsIosWatchKitExtension():
            assert self._IsBundle(), 'ios_watchkit_extension flag requires mac_bundle (target %s)' % self.spec['target_name']
            return 'com.apple.product-type.watchkit-extension'
        if self._IsIosWatchApp():
            assert self._IsBundle(), 'ios_watch_app flag requires mac_bundle (target %s)' % self.spec['target_name']
            return 'com.apple.product-type.application.watchapp'
        if self._IsXCUiTest():
            assert self._IsBundle(), 'mac_xcuitest_bundle flag requires mac_bundle (target %s)' % self.spec['target_name']
            return 'com.apple.product-type.bundle.ui-testing'
        if self._IsBundle():
            return {'executable': 'com.apple.product-type.application', 'loadable_module': 'com.apple.product-type.bundle', 'shared_library': 'com.apple.product-type.framework'}[self.spec['type']]
        else:
            return {'executable': 'com.apple.product-type.tool', 'loadable_module': 'com.apple.product-type.library.dynamic', 'shared_library': 'com.apple.product-type.library.dynamic', 'static_library': 'com.apple.product-type.library.static'}[self.spec['type']]

    def GetMachOType(self):
        """Returns the MACH_O_TYPE of this target."""
        if not self._IsBundle() and self.spec['type'] == 'executable':
            return ''
        return {'executable': 'mh_execute', 'static_library': 'staticlib', 'shared_library': 'mh_dylib', 'loadable_module': 'mh_bundle'}[self.spec['type']]

    def _GetBundleBinaryPath(self):
        """Returns the name of the bundle binary of by this target.
    E.g. Chromium.app/Contents/MacOS/Chromium. Only valid for bundles."""
        assert self._IsBundle()
        return os.path.join(self.GetBundleExecutableFolderPath(), self.GetExecutableName())

    def _GetStandaloneExecutableSuffix(self):
        if 'product_extension' in self.spec:
            return '.' + self.spec['product_extension']
        return {'executable': '', 'static_library': '.a', 'shared_library': '.dylib', 'loadable_module': '.so'}[self.spec['type']]

    def _GetStandaloneExecutablePrefix(self):
        return self.spec.get('product_prefix', {'executable': '', 'static_library': 'lib', 'shared_library': 'lib', 'loadable_module': ''}[self.spec['type']])

    def _GetStandaloneBinaryPath(self):
        """Returns the name of the non-bundle binary represented by this target.
    E.g. hello_world. Only valid for non-bundles."""
        assert not self._IsBundle()
        assert self.spec['type'] in ('executable', 'shared_library', 'static_library', 'loadable_module'), 'Unexpected type %s' % self.spec['type']
        target = self.spec['target_name']
        if self.spec['type'] == 'static_library':
            if target[:3] == 'lib':
                target = target[3:]
        elif self.spec['type'] in ('loadable_module', 'shared_library'):
            if target[:3] == 'lib':
                target = target[3:]
        target_prefix = self._GetStandaloneExecutablePrefix()
        target = self.spec.get('product_name', target)
        target_ext = self._GetStandaloneExecutableSuffix()
        return target_prefix + target + target_ext

    def GetExecutableName(self):
        """Returns the executable name of the bundle represented by this target.
    E.g. Chromium."""
        if self._IsBundle():
            return self.spec.get('product_name', self.spec['target_name'])
        else:
            return self._GetStandaloneBinaryPath()

    def GetExecutablePath(self):
        """Returns the qualified path to the primary executable of the bundle
    represented by this target. E.g. Chromium.app/Contents/MacOS/Chromium."""
        if self._IsBundle():
            return self._GetBundleBinaryPath()
        else:
            return self._GetStandaloneBinaryPath()

    def GetActiveArchs(self, configname):
        """Returns the architectures this target should be built for."""
        config_settings = self.xcode_settings[configname]
        xcode_archs_default = GetXcodeArchsDefault()
        return xcode_archs_default.ActiveArchs(config_settings.get('ARCHS'), config_settings.get('VALID_ARCHS'), config_settings.get('SDKROOT'))

    def _GetSdkVersionInfoItem(self, sdk, infoitem):
        try:
            return GetStdoutQuiet(['xcrun', '--sdk', sdk, infoitem])
        except GypError:
            pass

    def _SdkRoot(self, configname):
        if configname is None:
            configname = self.configname
        return self.GetPerConfigSetting('SDKROOT', configname, default='')

    def _XcodePlatformPath(self, configname=None):
        sdk_root = self._SdkRoot(configname)
        if sdk_root not in XcodeSettings._platform_path_cache:
            platform_path = self._GetSdkVersionInfoItem(sdk_root, '--show-sdk-platform-path')
            XcodeSettings._platform_path_cache[sdk_root] = platform_path
        return XcodeSettings._platform_path_cache[sdk_root]

    def _SdkPath(self, configname=None):
        sdk_root = self._SdkRoot(configname)
        if sdk_root.startswith('/'):
            return sdk_root
        return self._XcodeSdkPath(sdk_root)

    def _XcodeSdkPath(self, sdk_root):
        if sdk_root not in XcodeSettings._sdk_path_cache:
            sdk_path = self._GetSdkVersionInfoItem(sdk_root, '--show-sdk-path')
            XcodeSettings._sdk_path_cache[sdk_root] = sdk_path
            if sdk_root:
                XcodeSettings._sdk_root_cache[sdk_path] = sdk_root
        return XcodeSettings._sdk_path_cache[sdk_root]

    def _AppendPlatformVersionMinFlags(self, lst):
        self._Appendf(lst, 'MACOSX_DEPLOYMENT_TARGET', '-mmacosx-version-min=%s')
        if 'IPHONEOS_DEPLOYMENT_TARGET' in self._Settings():
            sdk_path_basename = os.path.basename(self._SdkPath())
            if sdk_path_basename.lower().startswith('iphonesimulator'):
                self._Appendf(lst, 'IPHONEOS_DEPLOYMENT_TARGET', '-mios-simulator-version-min=%s')
            else:
                self._Appendf(lst, 'IPHONEOS_DEPLOYMENT_TARGET', '-miphoneos-version-min=%s')

    def GetCflags(self, configname, arch=None):
        """Returns flags that need to be added to .c, .cc, .m, and .mm
    compilations."""
        self.configname = configname
        cflags = []
        sdk_root = self._SdkPath()
        if 'SDKROOT' in self._Settings() and sdk_root:
            cflags.append('-isysroot %s' % sdk_root)
        if self.header_map_path:
            cflags.append('-I%s' % self.header_map_path)
        if self._Test('CLANG_WARN_CONSTANT_CONVERSION', 'YES', default='NO'):
            cflags.append('-Wconstant-conversion')
        if self._Test('GCC_CHAR_IS_UNSIGNED_CHAR', 'YES', default='NO'):
            cflags.append('-funsigned-char')
        if self._Test('GCC_CW_ASM_SYNTAX', 'YES', default='YES'):
            cflags.append('-fasm-blocks')
        if 'GCC_DYNAMIC_NO_PIC' in self._Settings():
            if self._Settings()['GCC_DYNAMIC_NO_PIC'] == 'YES':
                cflags.append('-mdynamic-no-pic')
        else:
            pass
        if self._Test('GCC_ENABLE_PASCAL_STRINGS', 'YES', default='YES'):
            cflags.append('-mpascal-strings')
        self._Appendf(cflags, 'GCC_OPTIMIZATION_LEVEL', '-O%s', default='s')
        if self._Test('GCC_GENERATE_DEBUGGING_SYMBOLS', 'YES', default='YES'):
            dbg_format = self._Settings().get('DEBUG_INFORMATION_FORMAT', 'dwarf')
            if dbg_format == 'dwarf':
                cflags.append('-gdwarf-2')
            elif dbg_format == 'stabs':
                raise NotImplementedError('stabs debug format is not supported yet.')
            elif dbg_format == 'dwarf-with-dsym':
                cflags.append('-gdwarf-2')
            else:
                raise NotImplementedError('Unknown debug format %s' % dbg_format)
        if self._Settings().get('GCC_STRICT_ALIASING') == 'YES':
            cflags.append('-fstrict-aliasing')
        elif self._Settings().get('GCC_STRICT_ALIASING') == 'NO':
            cflags.append('-fno-strict-aliasing')
        if self._Test('GCC_SYMBOLS_PRIVATE_EXTERN', 'YES', default='NO'):
            cflags.append('-fvisibility=hidden')
        if self._Test('GCC_TREAT_WARNINGS_AS_ERRORS', 'YES', default='NO'):
            cflags.append('-Werror')
        if self._Test('GCC_WARN_ABOUT_MISSING_NEWLINE', 'YES', default='NO'):
            cflags.append('-Wnewline-eof')
        if self._Test('LLVM_LTO', 'YES', default='NO'):
            cflags.append('-flto')
        self._AppendPlatformVersionMinFlags(cflags)
        if self._Test('COPY_PHASE_STRIP', 'YES', default='NO'):
            self._WarnUnimplemented('COPY_PHASE_STRIP')
        self._WarnUnimplemented('GCC_DEBUGGING_SYMBOLS')
        self._WarnUnimplemented('GCC_ENABLE_OBJC_EXCEPTIONS')
        self._WarnUnimplemented('MACH_O_TYPE')
        self._WarnUnimplemented('PRODUCT_TYPE')
        if not gyp.common.CrossCompileRequested():
            if arch is not None:
                archs = [arch]
            else:
                assert self.configname
                archs = self.GetActiveArchs(self.configname)
            if len(archs) != 1:
                self._WarnUnimplemented('ARCHS')
                archs = ['i386']
            cflags.append('-arch ' + archs[0])
            if archs[0] in ('i386', 'x86_64'):
                if self._Test('GCC_ENABLE_SSE3_EXTENSIONS', 'YES', default='NO'):
                    cflags.append('-msse3')
                if self._Test('GCC_ENABLE_SUPPLEMENTAL_SSE3_INSTRUCTIONS', 'YES', default='NO'):
                    cflags.append('-mssse3')
                if self._Test('GCC_ENABLE_SSE41_EXTENSIONS', 'YES', default='NO'):
                    cflags.append('-msse4.1')
                if self._Test('GCC_ENABLE_SSE42_EXTENSIONS', 'YES', default='NO'):
                    cflags.append('-msse4.2')
        cflags += self._Settings().get('WARNING_CFLAGS', [])
        if self._IsXCTest():
            platform_root = self._XcodePlatformPath(configname)
            if platform_root:
                cflags.append('-F' + platform_root + '/Developer/Library/Frameworks/')
        if sdk_root:
            framework_root = sdk_root
        else:
            framework_root = ''
        config = self.spec['configurations'][self.configname]
        framework_dirs = config.get('mac_framework_dirs', [])
        for directory in framework_dirs:
            cflags.append('-F' + directory.replace('$(SDKROOT)', framework_root))
        self.configname = None
        return cflags

    def GetCflagsC(self, configname):
        """Returns flags that need to be added to .c, and .m compilations."""
        self.configname = configname
        cflags_c = []
        if self._Settings().get('GCC_C_LANGUAGE_STANDARD', '') == 'ansi':
            cflags_c.append('-ansi')
        else:
            self._Appendf(cflags_c, 'GCC_C_LANGUAGE_STANDARD', '-std=%s')
        cflags_c += self._Settings().get('OTHER_CFLAGS', [])
        self.configname = None
        return cflags_c

    def GetCflagsCC(self, configname):
        """Returns flags that need to be added to .cc, and .mm compilations."""
        self.configname = configname
        cflags_cc = []
        clang_cxx_language_standard = self._Settings().get('CLANG_CXX_LANGUAGE_STANDARD')
        if clang_cxx_language_standard:
            cflags_cc.append('-std=%s' % clang_cxx_language_standard)
        self._Appendf(cflags_cc, 'CLANG_CXX_LIBRARY', '-stdlib=%s')
        if self._Test('GCC_ENABLE_CPP_RTTI', 'NO', default='YES'):
            cflags_cc.append('-fno-rtti')
        if self._Test('GCC_ENABLE_CPP_EXCEPTIONS', 'NO', default='YES'):
            cflags_cc.append('-fno-exceptions')
        if self._Test('GCC_INLINES_ARE_PRIVATE_EXTERN', 'YES', default='NO'):
            cflags_cc.append('-fvisibility-inlines-hidden')
        if self._Test('GCC_THREADSAFE_STATICS', 'NO', default='YES'):
            cflags_cc.append('-fno-threadsafe-statics')
        if self._Test('GCC_WARN_ABOUT_INVALID_OFFSETOF_MACRO', 'NO', default='YES'):
            cflags_cc.append('-Wno-invalid-offsetof')
        other_ccflags = []
        for flag in self._Settings().get('OTHER_CPLUSPLUSFLAGS', ['$(inherited)']):
            if flag in ('$inherited', '$(inherited)', '${inherited}'):
                flag = '$OTHER_CFLAGS'
            if flag in ('$OTHER_CFLAGS', '$(OTHER_CFLAGS)', '${OTHER_CFLAGS}'):
                other_ccflags += self._Settings().get('OTHER_CFLAGS', [])
            else:
                other_ccflags.append(flag)
        cflags_cc += other_ccflags
        self.configname = None
        return cflags_cc

    def _AddObjectiveCGarbageCollectionFlags(self, flags):
        gc_policy = self._Settings().get('GCC_ENABLE_OBJC_GC', 'unsupported')
        if gc_policy == 'supported':
            flags.append('-fobjc-gc')
        elif gc_policy == 'required':
            flags.append('-fobjc-gc-only')

    def _AddObjectiveCARCFlags(self, flags):
        if self._Test('CLANG_ENABLE_OBJC_ARC', 'YES', default='NO'):
            flags.append('-fobjc-arc')

    def _AddObjectiveCMissingPropertySynthesisFlags(self, flags):
        if self._Test('CLANG_WARN_OBJC_MISSING_PROPERTY_SYNTHESIS', 'YES', default='NO'):
            flags.append('-Wobjc-missing-property-synthesis')

    def GetCflagsObjC(self, configname):
        """Returns flags that need to be added to .m compilations."""
        self.configname = configname
        cflags_objc = []
        self._AddObjectiveCGarbageCollectionFlags(cflags_objc)
        self._AddObjectiveCARCFlags(cflags_objc)
        self._AddObjectiveCMissingPropertySynthesisFlags(cflags_objc)
        self.configname = None
        return cflags_objc

    def GetCflagsObjCC(self, configname):
        """Returns flags that need to be added to .mm compilations."""
        self.configname = configname
        cflags_objcc = []
        self._AddObjectiveCGarbageCollectionFlags(cflags_objcc)
        self._AddObjectiveCARCFlags(cflags_objcc)
        self._AddObjectiveCMissingPropertySynthesisFlags(cflags_objcc)
        if self._Test('GCC_OBJC_CALL_CXX_CDTORS', 'YES', default='NO'):
            cflags_objcc.append('-fobjc-call-cxx-cdtors')
        self.configname = None
        return cflags_objcc

    def GetInstallNameBase(self):
        """Return DYLIB_INSTALL_NAME_BASE for this target."""
        if self.spec['type'] != 'shared_library' and (self.spec['type'] != 'loadable_module' or self._IsBundle()):
            return None
        install_base = self.GetPerTargetSetting('DYLIB_INSTALL_NAME_BASE', default='/Library/Frameworks' if self._IsBundle() else '/usr/local/lib')
        return install_base

    def _StandardizePath(self, path):
        """Do :standardizepath processing for path."""
        if '/' in path:
            prefix, rest = ('', path)
            if path.startswith('@'):
                prefix, rest = path.split('/', 1)
            rest = os.path.normpath(rest)
            path = os.path.join(prefix, rest)
        return path

    def GetInstallName(self):
        """Return LD_DYLIB_INSTALL_NAME for this target."""
        if self.spec['type'] != 'shared_library' and (self.spec['type'] != 'loadable_module' or self._IsBundle()):
            return None
        default_install_name = '$(DYLIB_INSTALL_NAME_BASE:standardizepath)/$(EXECUTABLE_PATH)'
        install_name = self.GetPerTargetSetting('LD_DYLIB_INSTALL_NAME', default=default_install_name)
        if '$' in install_name:
            assert install_name in ('$(DYLIB_INSTALL_NAME_BASE:standardizepath)/$(WRAPPER_NAME)/$(PRODUCT_NAME)', default_install_name), "Variables in LD_DYLIB_INSTALL_NAME are not generally supported yet in target '%s' (got '%s')" % (self.spec['target_name'], install_name)
            install_name = install_name.replace('$(DYLIB_INSTALL_NAME_BASE:standardizepath)', self._StandardizePath(self.GetInstallNameBase()))
            if self._IsBundle():
                install_name = install_name.replace('$(WRAPPER_NAME)', self.GetWrapperName())
                install_name = install_name.replace('$(PRODUCT_NAME)', self.GetProductName())
            else:
                assert '$(WRAPPER_NAME)' not in install_name
                assert '$(PRODUCT_NAME)' not in install_name
            install_name = install_name.replace('$(EXECUTABLE_PATH)', self.GetExecutablePath())
        return install_name

    def _MapLinkerFlagFilename(self, ldflag, gyp_to_build_path):
        """Checks if ldflag contains a filename and if so remaps it from
    gyp-directory-relative to build-directory-relative."""
        LINKER_FILE = '(\\S+)'
        WORD = '\\S+'
        linker_flags = [['-exported_symbols_list', LINKER_FILE], ['-unexported_symbols_list', LINKER_FILE], ['-reexported_symbols_list', LINKER_FILE], ['-sectcreate', WORD, WORD, LINKER_FILE]]
        for flag_pattern in linker_flags:
            regex = re.compile('(?:-Wl,)?' + '[ ,]'.join(flag_pattern))
            m = regex.match(ldflag)
            if m:
                ldflag = ldflag[:m.start(1)] + gyp_to_build_path(m.group(1)) + ldflag[m.end(1):]
        if ldflag.startswith('-L'):
            ldflag = '-L' + gyp_to_build_path(ldflag[len('-L'):])
        return ldflag

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

    def GetLibtoolflags(self, configname):
        """Returns flags that need to be passed to the static linker.

    Args:
        configname: The name of the configuration to get ld flags for.
    """
        self.configname = configname
        libtoolflags = []
        for libtoolflag in self._Settings().get('OTHER_LDFLAGS', []):
            libtoolflags.append(libtoolflag)
        self.configname = None
        return libtoolflags

    def GetPerTargetSettings(self):
        """Gets a list of all the per-target settings. This will only fetch keys
    whose values are the same across all configurations."""
        first_pass = True
        result = {}
        for configname in sorted(self.xcode_settings.keys()):
            if first_pass:
                result = dict(self.xcode_settings[configname])
                first_pass = False
            else:
                for key, value in self.xcode_settings[configname].items():
                    if key not in result:
                        continue
                    elif result[key] != value:
                        del result[key]
        return result

    def GetPerConfigSetting(self, setting, configname, default=None):
        if configname in self.xcode_settings:
            return self.xcode_settings[configname].get(setting, default)
        else:
            return self.GetPerTargetSetting(setting, default)

    def GetPerTargetSetting(self, setting, default=None):
        """Tries to get xcode_settings.setting from spec. Assumes that the setting
       has the same value in all configurations and throws otherwise."""
        is_first_pass = True
        result = None
        for configname in sorted(self.xcode_settings.keys()):
            if is_first_pass:
                result = self.xcode_settings[configname].get(setting, None)
                is_first_pass = False
            else:
                assert result == self.xcode_settings[configname].get(setting, None), "Expected per-target setting for '%s', got per-config setting (target %s)" % (setting, self.spec['target_name'])
        if result is None:
            return default
        return result

    def _GetStripPostbuilds(self, configname, output_binary, quiet):
        """Returns a list of shell commands that contain the shell commands
    necessary to strip this target's binary. These should be run as postbuilds
    before the actual postbuilds run."""
        self.configname = configname
        result = []
        if self._Test('DEPLOYMENT_POSTPROCESSING', 'YES', default='NO') and self._Test('STRIP_INSTALLED_PRODUCT', 'YES', default='NO'):
            default_strip_style = 'debugging'
            if (self.spec['type'] == 'loadable_module' or self._IsIosAppExtension()) and self._IsBundle():
                default_strip_style = 'non-global'
            elif self.spec['type'] == 'executable':
                default_strip_style = 'all'
            strip_style = self._Settings().get('STRIP_STYLE', default_strip_style)
            strip_flags = {'all': '', 'non-global': '-x', 'debugging': '-S'}[strip_style]
            explicit_strip_flags = self._Settings().get('STRIPFLAGS', '')
            if explicit_strip_flags:
                strip_flags += ' ' + _NormalizeEnvVarReferences(explicit_strip_flags)
            if not quiet:
                result.append('echo STRIP\\(%s\\)' % self.spec['target_name'])
            result.append(f'strip {strip_flags} {output_binary}')
        self.configname = None
        return result

    def _GetDebugInfoPostbuilds(self, configname, output, output_binary, quiet):
        """Returns a list of shell commands that contain the shell commands
    necessary to massage this target's debug information. These should be run
    as postbuilds before the actual postbuilds run."""
        self.configname = configname
        result = []
        if self._Test('GCC_GENERATE_DEBUGGING_SYMBOLS', 'YES', default='YES') and self._Test('DEBUG_INFORMATION_FORMAT', 'dwarf-with-dsym', default='dwarf') and (self.spec['type'] != 'static_library'):
            if not quiet:
                result.append('echo DSYMUTIL\\(%s\\)' % self.spec['target_name'])
            result.append('dsymutil {} -o {}'.format(output_binary, output + '.dSYM'))
        self.configname = None
        return result

    def _GetTargetPostbuilds(self, configname, output, output_binary, quiet=False):
        """Returns a list of shell commands that contain the shell commands
    to run as postbuilds for this target, before the actual postbuilds."""
        return self._GetDebugInfoPostbuilds(configname, output, output_binary, quiet) + self._GetStripPostbuilds(configname, output_binary, quiet)

    def _GetIOSPostbuilds(self, configname, output_binary):
        """Return a shell command to codesign the iOS output binary so it can
    be deployed to a device.  This should be run as the very last step of the
    build."""
        if not (self.isIOS and (self.spec['type'] == 'executable' or self._IsXCTest()) or self.IsIosFramework()):
            return []
        postbuilds = []
        product_name = self.GetFullProductName()
        settings = self.xcode_settings[configname]
        if self._IsXCTest():
            source = os.path.join('${BUILT_PRODUCTS_DIR}', product_name)
            test_host = os.path.dirname(settings.get('TEST_HOST'))
            xctest_destination = os.path.join(test_host, 'PlugIns', product_name)
            postbuilds.extend([f'ditto {source} {xctest_destination}'])
        key = self._GetIOSCodeSignIdentityKey(settings)
        if not key:
            return postbuilds
        unimpl = ['OTHER_CODE_SIGN_FLAGS']
        unimpl = set(unimpl) & set(self.xcode_settings[configname].keys())
        if unimpl:
            print('Warning: Some codesign keys not implemented, ignoring: %s' % ', '.join(sorted(unimpl)))
        if self._IsXCTest():
            test_host = os.path.dirname(settings.get('TEST_HOST'))
            frameworks_dir = os.path.join(test_host, 'Frameworks')
            platform_root = self._XcodePlatformPath(configname)
            frameworks = ['Developer/Library/PrivateFrameworks/IDEBundleInjection.framework', 'Developer/Library/Frameworks/XCTest.framework']
            for framework in frameworks:
                source = os.path.join(platform_root, framework)
                destination = os.path.join(frameworks_dir, os.path.basename(framework))
                postbuilds.extend([f'ditto {source} {destination}'])
                postbuilds.extend(['%s code-sign-bundle "%s" "%s" "%s" "%s" %s' % (os.path.join('${TARGET_BUILD_DIR}', 'gyp-mac-tool'), key, settings.get('CODE_SIGN_ENTITLEMENTS', ''), settings.get('PROVISIONING_PROFILE', ''), destination, True)])
            plugin_dir = os.path.join(test_host, 'PlugIns')
            targets = [os.path.join(plugin_dir, product_name), test_host]
            for target in targets:
                postbuilds.extend(['%s code-sign-bundle "%s" "%s" "%s" "%s" %s' % (os.path.join('${TARGET_BUILD_DIR}', 'gyp-mac-tool'), key, settings.get('CODE_SIGN_ENTITLEMENTS', ''), settings.get('PROVISIONING_PROFILE', ''), target, True)])
        postbuilds.extend(['%s code-sign-bundle "%s" "%s" "%s" "%s" %s' % (os.path.join('${TARGET_BUILD_DIR}', 'gyp-mac-tool'), key, settings.get('CODE_SIGN_ENTITLEMENTS', ''), settings.get('PROVISIONING_PROFILE', ''), os.path.join('${BUILT_PRODUCTS_DIR}', product_name), False)])
        return postbuilds

    def _GetIOSCodeSignIdentityKey(self, settings):
        identity = settings.get('CODE_SIGN_IDENTITY')
        if not identity:
            return None
        if identity not in XcodeSettings._codesigning_key_cache:
            output = subprocess.check_output(['security', 'find-identity', '-p', 'codesigning', '-v'])
            for line in output.splitlines():
                if identity in line:
                    fingerprint = line.split()[1]
                    cache = XcodeSettings._codesigning_key_cache
                    assert identity not in cache or fingerprint == cache[identity], 'Multiple codesigning fingerprints for identity: %s' % identity
                    XcodeSettings._codesigning_key_cache[identity] = fingerprint
        return XcodeSettings._codesigning_key_cache.get(identity, '')

    def AddImplicitPostbuilds(self, configname, output, output_binary, postbuilds=[], quiet=False):
        """Returns a list of shell commands that should run before and after
    |postbuilds|."""
        assert output_binary is not None
        pre = self._GetTargetPostbuilds(configname, output, output_binary, quiet)
        post = self._GetIOSPostbuilds(configname, output_binary)
        return pre + postbuilds + post

    def _AdjustLibrary(self, library, config_name=None):
        if library.endswith('.framework'):
            l_flag = '-framework ' + os.path.splitext(os.path.basename(library))[0]
        else:
            m = self.library_re.match(library)
            if m:
                l_flag = '-l' + m.group(1)
            else:
                l_flag = library
        sdk_root = self._SdkPath(config_name)
        if not sdk_root:
            sdk_root = ''
        library = l_flag.replace('$(SDKROOT)', sdk_root)
        if l_flag.startswith('$(SDKROOT)'):
            basename, ext = os.path.splitext(library)
            if ext == '.dylib' and (not os.path.exists(library)):
                tbd_library = basename + '.tbd'
                if os.path.exists(tbd_library):
                    library = tbd_library
        return library

    def AdjustLibraries(self, libraries, config_name=None):
        """Transforms entries like 'Cocoa.framework' in libraries into entries like
    '-framework Cocoa', 'libcrypto.dylib' into '-lcrypto', etc.
    """
        libraries = [self._AdjustLibrary(library, config_name) for library in libraries]
        return libraries

    def _BuildMachineOSBuild(self):
        return GetStdout(['sw_vers', '-buildVersion'])

    def _XcodeIOSDeviceFamily(self, configname):
        family = self.xcode_settings[configname].get('TARGETED_DEVICE_FAMILY', '1')
        return [int(x) for x in family.split(',')]

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

    def _DefaultSdkRoot(self):
        """Returns the default SDKROOT to use.

    Prior to version 5.0.0, if SDKROOT was not explicitly set in the Xcode
    project, then the environment variable was empty. Starting with this
    version, Xcode uses the name of the newest SDK installed.
    """
        xcode_version, _ = XcodeVersion()
        if xcode_version < '0500':
            return ''
        default_sdk_path = self._XcodeSdkPath('')
        default_sdk_root = XcodeSettings._sdk_root_cache.get(default_sdk_path)
        if default_sdk_root:
            return default_sdk_root
        try:
            all_sdks = GetStdout(['xcodebuild', '-showsdks'])
        except GypError:
            return ''
        for line in all_sdks.splitlines():
            items = line.split()
            if len(items) >= 3 and items[-2] == '-sdk':
                sdk_root = items[-1]
                sdk_path = self._XcodeSdkPath(sdk_root)
                if sdk_path == default_sdk_path:
                    return sdk_root
        return ''