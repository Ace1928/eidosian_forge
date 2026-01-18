import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
class EnvironmentInfo:
    """
    Return environment variables for specified Microsoft Visual C++ version
    and platform : Lib, Include, Path and libpath.

    This function is compatible with Microsoft Visual C++ 9.0 to 14.X.

    Script created by analysing Microsoft environment configuration files like
    "vcvars[...].bat", "SetEnv.Cmd", "vcbuildtools.bat", ...

    Parameters
    ----------
    arch: str
        Target architecture.
    vc_ver: float
        Required Microsoft Visual C++ version. If not set, autodetect the last
        version.
    vc_min_ver: float
        Minimum Microsoft Visual C++ version.
    """

    def __init__(self, arch, vc_ver=None, vc_min_ver=0):
        self.pi = PlatformInfo(arch)
        self.ri = RegistryInfo(self.pi)
        self.si = SystemInfo(self.ri, vc_ver)
        if self.vc_ver < vc_min_ver:
            err = 'No suitable Microsoft Visual C++ version found'
            raise distutils.errors.DistutilsPlatformError(err)

    @property
    def vs_ver(self):
        """
        Microsoft Visual Studio.

        Return
        ------
        float
            version
        """
        return self.si.vs_ver

    @property
    def vc_ver(self):
        """
        Microsoft Visual C++ version.

        Return
        ------
        float
            version
        """
        return self.si.vc_ver

    @property
    def VSTools(self):
        """
        Microsoft Visual Studio Tools.

        Return
        ------
        list of str
            paths
        """
        paths = ['Common7\\IDE', 'Common7\\Tools']
        if self.vs_ver >= 14.0:
            arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            paths += ['Common7\\IDE\\CommonExtensions\\Microsoft\\TestWindow']
            paths += ['Team Tools\\Performance Tools']
            paths += ['Team Tools\\Performance Tools%s' % arch_subdir]
        return [join(self.si.VSInstallDir, path) for path in paths]

    @property
    def VCIncludes(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Includes.

        Return
        ------
        list of str
            paths
        """
        return [join(self.si.VCInstallDir, 'Include'), join(self.si.VCInstallDir, 'ATLMFC\\Include')]

    @property
    def VCLibraries(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver >= 15.0:
            arch_subdir = self.pi.target_dir(x64=True)
        else:
            arch_subdir = self.pi.target_dir(hidex86=True)
        paths = ['Lib%s' % arch_subdir, 'ATLMFC\\Lib%s' % arch_subdir]
        if self.vs_ver >= 14.0:
            paths += ['Lib\\store%s' % arch_subdir]
        return [join(self.si.VCInstallDir, path) for path in paths]

    @property
    def VCStoreRefs(self):
        """
        Microsoft Visual C++ store references Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []
        return [join(self.si.VCInstallDir, 'Lib\\store\\references')]

    @property
    def VCTools(self):
        """
        Microsoft Visual C++ Tools.

        Return
        ------
        list of str
            paths
        """
        si = self.si
        tools = [join(si.VCInstallDir, 'VCPackages')]
        forcex86 = True if self.vs_ver <= 10.0 else False
        arch_subdir = self.pi.cross_dir(forcex86)
        if arch_subdir:
            tools += [join(si.VCInstallDir, 'Bin%s' % arch_subdir)]
        if self.vs_ver == 14.0:
            path = 'Bin%s' % self.pi.current_dir(hidex86=True)
            tools += [join(si.VCInstallDir, path)]
        elif self.vs_ver >= 15.0:
            host_dir = 'bin\\HostX86%s' if self.pi.current_is_x86() else 'bin\\HostX64%s'
            tools += [join(si.VCInstallDir, host_dir % self.pi.target_dir(x64=True))]
            if self.pi.current_cpu != self.pi.target_cpu:
                tools += [join(si.VCInstallDir, host_dir % self.pi.current_dir(x64=True))]
        else:
            tools += [join(si.VCInstallDir, 'Bin')]
        return tools

    @property
    def OSLibraries(self):
        """
        Microsoft Windows SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver <= 10.0:
            arch_subdir = self.pi.target_dir(hidex86=True, x64=True)
            return [join(self.si.WindowsSdkDir, 'Lib%s' % arch_subdir)]
        else:
            arch_subdir = self.pi.target_dir(x64=True)
            lib = join(self.si.WindowsSdkDir, 'lib')
            libver = self._sdk_subdir
            return [join(lib, '%sum%s' % (libver, arch_subdir))]

    @property
    def OSIncludes(self):
        """
        Microsoft Windows SDK Include.

        Return
        ------
        list of str
            paths
        """
        include = join(self.si.WindowsSdkDir, 'include')
        if self.vs_ver <= 10.0:
            return [include, join(include, 'gl')]
        else:
            if self.vs_ver >= 14.0:
                sdkver = self._sdk_subdir
            else:
                sdkver = ''
            return [join(include, '%sshared' % sdkver), join(include, '%sum' % sdkver), join(include, '%swinrt' % sdkver)]

    @property
    def OSLibpath(self):
        """
        Microsoft Windows SDK Libraries Paths.

        Return
        ------
        list of str
            paths
        """
        ref = join(self.si.WindowsSdkDir, 'References')
        libpath = []
        if self.vs_ver <= 9.0:
            libpath += self.OSLibraries
        if self.vs_ver >= 11.0:
            libpath += [join(ref, 'CommonConfiguration\\Neutral')]
        if self.vs_ver >= 14.0:
            libpath += [ref, join(self.si.WindowsSdkDir, 'UnionMetadata'), join(ref, 'Windows.Foundation.UniversalApiContract', '1.0.0.0'), join(ref, 'Windows.Foundation.FoundationContract', '1.0.0.0'), join(ref, 'Windows.Networking.Connectivity.WwanContract', '1.0.0.0'), join(self.si.WindowsSdkDir, 'ExtensionSDKs', 'Microsoft.VCLibs', '%0.1f' % self.vs_ver, 'References', 'CommonConfiguration', 'neutral')]
        return libpath

    @property
    def SdkTools(self):
        """
        Microsoft Windows SDK Tools.

        Return
        ------
        list of str
            paths
        """
        return list(self._sdk_tools())

    def _sdk_tools(self):
        """
        Microsoft Windows SDK Tools paths generator.

        Return
        ------
        generator of str
            paths
        """
        if self.vs_ver < 15.0:
            bin_dir = 'Bin' if self.vs_ver <= 11.0 else 'Bin\\x86'
            yield join(self.si.WindowsSdkDir, bin_dir)
        if not self.pi.current_is_x86():
            arch_subdir = self.pi.current_dir(x64=True)
            path = 'Bin%s' % arch_subdir
            yield join(self.si.WindowsSdkDir, path)
        if self.vs_ver in (10.0, 11.0):
            if self.pi.target_is_x86():
                arch_subdir = ''
            else:
                arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            path = 'Bin\\NETFX 4.0 Tools%s' % arch_subdir
            yield join(self.si.WindowsSdkDir, path)
        elif self.vs_ver >= 15.0:
            path = join(self.si.WindowsSdkDir, 'Bin')
            arch_subdir = self.pi.current_dir(x64=True)
            sdkver = self.si.WindowsSdkLastVersion
            yield join(path, '%s%s' % (sdkver, arch_subdir))
        if self.si.WindowsSDKExecutablePath:
            yield self.si.WindowsSDKExecutablePath

    @property
    def _sdk_subdir(self):
        """
        Microsoft Windows SDK version subdir.

        Return
        ------
        str
            subdir
        """
        ucrtver = self.si.WindowsSdkLastVersion
        return '%s\\' % ucrtver if ucrtver else ''

    @property
    def SdkSetup(self):
        """
        Microsoft Windows SDK Setup.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver > 9.0:
            return []
        return [join(self.si.WindowsSdkDir, 'Setup')]

    @property
    def FxTools(self):
        """
        Microsoft .NET Framework Tools.

        Return
        ------
        list of str
            paths
        """
        pi = self.pi
        si = self.si
        if self.vs_ver <= 10.0:
            include32 = True
            include64 = not pi.target_is_x86() and (not pi.current_is_x86())
        else:
            include32 = pi.target_is_x86() or pi.current_is_x86()
            include64 = pi.current_cpu == 'amd64' or pi.target_cpu == 'amd64'
        tools = []
        if include32:
            tools += [join(si.FrameworkDir32, ver) for ver in si.FrameworkVersion32]
        if include64:
            tools += [join(si.FrameworkDir64, ver) for ver in si.FrameworkVersion64]
        return tools

    @property
    def NetFxSDKLibraries(self):
        """
        Microsoft .Net Framework SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []
        arch_subdir = self.pi.target_dir(x64=True)
        return [join(self.si.NetFxSdkDir, 'lib\\um%s' % arch_subdir)]

    @property
    def NetFxSDKIncludes(self):
        """
        Microsoft .Net Framework SDK Includes.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []
        return [join(self.si.NetFxSdkDir, 'include\\um')]

    @property
    def VsTDb(self):
        """
        Microsoft Visual Studio Team System Database.

        Return
        ------
        list of str
            paths
        """
        return [join(self.si.VSInstallDir, 'VSTSDB\\Deploy')]

    @property
    def MSBuild(self):
        """
        Microsoft Build Engine.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 12.0:
            return []
        elif self.vs_ver < 15.0:
            base_path = self.si.ProgramFilesx86
            arch_subdir = self.pi.current_dir(hidex86=True)
        else:
            base_path = self.si.VSInstallDir
            arch_subdir = ''
        path = 'MSBuild\\%0.1f\\bin%s' % (self.vs_ver, arch_subdir)
        build = [join(base_path, path)]
        if self.vs_ver >= 15.0:
            build += [join(base_path, path, 'Roslyn')]
        return build

    @property
    def HTMLHelpWorkshop(self):
        """
        Microsoft HTML Help Workshop.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 11.0:
            return []
        return [join(self.si.ProgramFilesx86, 'HTML Help Workshop')]

    @property
    def UCRTLibraries(self):
        """
        Microsoft Universal C Runtime SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []
        arch_subdir = self.pi.target_dir(x64=True)
        lib = join(self.si.UniversalCRTSdkDir, 'lib')
        ucrtver = self._ucrt_subdir
        return [join(lib, '%sucrt%s' % (ucrtver, arch_subdir))]

    @property
    def UCRTIncludes(self):
        """
        Microsoft Universal C Runtime SDK Include.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []
        include = join(self.si.UniversalCRTSdkDir, 'include')
        return [join(include, '%sucrt' % self._ucrt_subdir)]

    @property
    def _ucrt_subdir(self):
        """
        Microsoft Universal C Runtime SDK version subdir.

        Return
        ------
        str
            subdir
        """
        ucrtver = self.si.UniversalCRTSdkLastVersion
        return '%s\\' % ucrtver if ucrtver else ''

    @property
    def FSharp(self):
        """
        Microsoft Visual F#.

        Return
        ------
        list of str
            paths
        """
        if 11.0 > self.vs_ver > 12.0:
            return []
        return [self.si.FSharpInstallDir]

    @property
    def VCRuntimeRedist(self):
        """
        Microsoft Visual C++ runtime redistributable dll.

        Return
        ------
        str
            path
        """
        vcruntime = 'vcruntime%d0.dll' % self.vc_ver
        arch_subdir = self.pi.target_dir(x64=True).strip('\\')
        prefixes = []
        tools_path = self.si.VCInstallDir
        redist_path = dirname(tools_path.replace('\\Tools', '\\Redist'))
        if isdir(redist_path):
            redist_path = join(redist_path, listdir(redist_path)[-1])
            prefixes += [redist_path, join(redist_path, 'onecore')]
        prefixes += [join(tools_path, 'redist')]
        crt_dirs = ('Microsoft.VC%d.CRT' % (self.vc_ver * 10), 'Microsoft.VC%d.CRT' % (int(self.vs_ver) * 10))
        for prefix, crt_dir in itertools.product(prefixes, crt_dirs):
            path = join(prefix, arch_subdir, crt_dir, vcruntime)
            if isfile(path):
                return path

    def return_env(self, exists=True):
        """
        Return environment dict.

        Parameters
        ----------
        exists: bool
            It True, only return existing paths.

        Return
        ------
        dict
            environment
        """
        env = dict(include=self._build_paths('include', [self.VCIncludes, self.OSIncludes, self.UCRTIncludes, self.NetFxSDKIncludes], exists), lib=self._build_paths('lib', [self.VCLibraries, self.OSLibraries, self.FxTools, self.UCRTLibraries, self.NetFxSDKLibraries], exists), libpath=self._build_paths('libpath', [self.VCLibraries, self.FxTools, self.VCStoreRefs, self.OSLibpath], exists), path=self._build_paths('path', [self.VCTools, self.VSTools, self.VsTDb, self.SdkTools, self.SdkSetup, self.FxTools, self.MSBuild, self.HTMLHelpWorkshop, self.FSharp], exists))
        if self.vs_ver >= 14 and isfile(self.VCRuntimeRedist):
            env['py_vcruntime_redist'] = self.VCRuntimeRedist
        return env

    def _build_paths(self, name, spec_path_lists, exists):
        """
        Given an environment variable name and specified paths,
        return a pathsep-separated string of paths containing
        unique, extant, directories from those paths and from
        the environment variable. Raise an error if no paths
        are resolved.

        Parameters
        ----------
        name: str
            Environment variable name
        spec_path_lists: list of str
            Paths
        exists: bool
            It True, only return existing paths.

        Return
        ------
        str
            Pathsep-separated paths
        """
        spec_paths = itertools.chain.from_iterable(spec_path_lists)
        env_paths = environ.get(name, '').split(pathsep)
        paths = itertools.chain(spec_paths, env_paths)
        extant_paths = list(filter(isdir, paths)) if exists else paths
        if not extant_paths:
            msg = '%s environment variable is empty' % name.upper()
            raise distutils.errors.DistutilsPlatformError(msg)
        unique_paths = unique_everseen(extant_paths)
        return pathsep.join(unique_paths)