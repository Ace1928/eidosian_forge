from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
class Vs2010Backend(backends.Backend):
    name = 'vs2010'

    def __init__(self, build: T.Optional[build.Build], interpreter: T.Optional[Interpreter], gen_lite: bool=False):
        super().__init__(build, interpreter)
        self.project_file_version = '10.0.30319.1'
        self.sln_file_version = '11.00'
        self.sln_version_comment = '2010'
        self.platform_toolset = None
        self.vs_version = '2010'
        self.windows_target_platform_version = None
        self.subdirs = {}
        self.handled_target_deps = {}
        self.gen_lite = gen_lite

    def get_target_private_dir(self, target):
        return os.path.join(self.get_target_dir(target), target.get_id())

    def generate_genlist_for_target(self, genlist: T.Union[build.GeneratedList, build.CustomTarget, build.CustomTargetIndex], target: build.BuildTarget, parent_node: ET.Element, generator_output_files: T.List[str], custom_target_include_dirs: T.List[str], custom_target_output_files: T.List[str]) -> None:
        if isinstance(genlist, build.GeneratedList):
            for x in genlist.depends:
                self.generate_genlist_for_target(x, target, parent_node, [], [], [])
        target_private_dir = self.relpath(self.get_target_private_dir(target), self.get_target_dir(target))
        down = self.target_to_build_root(target)
        if isinstance(genlist, (build.CustomTarget, build.CustomTargetIndex)):
            for i in genlist.get_outputs():
                ipath = os.path.join(down, self.get_target_dir(genlist), i)
                custom_target_output_files.append(ipath)
            idir = self.relpath(self.get_target_dir(genlist), self.get_target_dir(target))
            if idir not in custom_target_include_dirs:
                custom_target_include_dirs.append(idir)
        else:
            generator = genlist.get_generator()
            exe = generator.get_exe()
            infilelist = genlist.get_inputs()
            outfilelist = genlist.get_outputs()
            source_dir = os.path.join(down, self.build_to_src, genlist.subdir)
            idgroup = ET.SubElement(parent_node, 'ItemGroup')
            samelen = len(infilelist) == len(outfilelist)
            for i, curfile in enumerate(infilelist):
                if samelen:
                    sole_output = os.path.join(target_private_dir, outfilelist[i])
                else:
                    sole_output = ''
                infilename = os.path.join(down, curfile.rel_to_builddir(self.build_to_src, target_private_dir))
                deps = self.get_target_depend_files(genlist, True)
                base_args = generator.get_arglist(infilename)
                outfiles_rel = genlist.get_outputs_for(curfile)
                outfiles = [os.path.join(target_private_dir, of) for of in outfiles_rel]
                generator_output_files += outfiles
                args = [x.replace('@INPUT@', infilename).replace('@OUTPUT@', sole_output) for x in base_args]
                args = self.replace_outputs(args, target_private_dir, outfiles_rel)
                args = [x.replace('@SOURCE_DIR@', self.environment.get_source_dir()).replace('@BUILD_DIR@', target_private_dir) for x in args]
                args = [x.replace('@CURRENT_SOURCE_DIR@', source_dir) for x in args]
                args = [x.replace('@SOURCE_ROOT@', self.environment.get_source_dir()).replace('@BUILD_ROOT@', self.environment.get_build_dir()) for x in args]
                args = [x.replace('\\', '/') for x in args]
                tdir_abs = os.path.join(self.environment.get_build_dir(), self.get_target_dir(target))
                cmd, _ = self.as_meson_exe_cmdline(exe, self.replace_extra_args(args, genlist), workdir=tdir_abs, capture=outfiles[0] if generator.capture else None, force_serialize=True, env=genlist.env)
                deps = cmd[-1:] + deps
                abs_pdir = os.path.join(self.environment.get_build_dir(), self.get_target_dir(target))
                os.makedirs(abs_pdir, exist_ok=True)
                cbs = ET.SubElement(idgroup, 'CustomBuild', Include=infilename)
                ET.SubElement(cbs, 'Command').text = ' '.join(self.quote_arguments(cmd))
                ET.SubElement(cbs, 'Outputs').text = ';'.join(outfiles)
                ET.SubElement(cbs, 'AdditionalInputs').text = ';'.join(deps)

    def generate_custom_generator_commands(self, target, parent_node):
        generator_output_files = []
        custom_target_include_dirs = []
        custom_target_output_files = []
        for genlist in target.get_generated_sources():
            self.generate_genlist_for_target(genlist, target, parent_node, generator_output_files, custom_target_include_dirs, custom_target_output_files)
        return (generator_output_files, custom_target_output_files, custom_target_include_dirs)

    def generate(self, capture: bool=False, vslite_ctx: dict=None) -> T.Optional[dict]:
        if capture:
            raise MesonBugException("We do not expect any vs backend to generate with 'capture = True'")
        host_machine = self.environment.machines.host.cpu_family
        if host_machine in {'64', 'x86_64'}:
            target_system = self.environment.machines.host.system
            if detect_microsoft_gdk(target_system):
                self.platform = target_system
            else:
                self.platform = 'x64'
        elif host_machine == 'x86':
            self.platform = 'Win32'
        elif host_machine in {'aarch64', 'arm64'}:
            target_cpu = self.environment.machines.host.cpu
            if target_cpu == 'arm64ec':
                self.platform = 'arm64ec'
            else:
                self.platform = 'arm64'
        elif 'arm' in host_machine.lower():
            self.platform = 'ARM'
        else:
            raise MesonException('Unsupported Visual Studio platform: ' + host_machine)
        build_machine = self.environment.machines.build.cpu_family
        if build_machine in {'64', 'x86_64'}:
            self.build_platform = 'x64'
        elif build_machine == 'x86':
            self.build_platform = 'Win32'
        elif build_machine in {'aarch64', 'arm64'}:
            target_cpu = self.environment.machines.build.cpu
            if target_cpu == 'arm64ec':
                self.build_platform = 'arm64ec'
            else:
                self.build_platform = 'arm64'
        elif 'arm' in build_machine.lower():
            self.build_platform = 'ARM'
        else:
            raise MesonException('Unsupported Visual Studio platform: ' + build_machine)
        self.buildtype = self.environment.coredata.get_option(OptionKey('buildtype'))
        self.optimization = self.environment.coredata.get_option(OptionKey('optimization'))
        self.debug = self.environment.coredata.get_option(OptionKey('debug'))
        try:
            self.sanitize = self.environment.coredata.get_option(OptionKey('b_sanitize'))
        except MesonException:
            self.sanitize = 'none'
        sln_filename = os.path.join(self.environment.get_build_dir(), self.build.project_name + '.sln')
        projlist = self.generate_projects(vslite_ctx)
        self.gen_testproj()
        self.gen_installproj()
        self.gen_regenproj()
        self.generate_solution(sln_filename, projlist)
        self.generate_regen_info()
        Vs2010Backend.touch_regen_timestamp(self.environment.get_build_dir())

    @staticmethod
    def get_regen_stampfile(build_dir: str) -> None:
        return os.path.join(os.path.join(build_dir, Environment.private_dir), 'regen.stamp')

    @staticmethod
    def touch_regen_timestamp(build_dir: str) -> None:
        with open(Vs2010Backend.get_regen_stampfile(build_dir), 'w', encoding='utf-8'):
            pass

    def get_vcvars_command(self):
        has_arch_values = 'VSCMD_ARG_TGT_ARCH' in os.environ and 'VSCMD_ARG_HOST_ARCH' in os.environ
        if 'VCINSTALLDIR' in os.environ:
            vs_version = os.environ['VisualStudioVersion'] if 'VisualStudioVersion' in os.environ else None
            relative_path = 'Auxiliary\\Build\\' if vs_version is not None and vs_version >= '15.0' else ''
            script_path = os.environ['VCINSTALLDIR'] + relative_path + 'vcvarsall.bat'
            if os.path.exists(script_path):
                if has_arch_values:
                    target_arch = os.environ['VSCMD_ARG_TGT_ARCH']
                    host_arch = os.environ['VSCMD_ARG_HOST_ARCH']
                else:
                    target_arch = os.environ.get('Platform', 'x86')
                    host_arch = target_arch
                arch = host_arch + '_' + target_arch if host_arch != target_arch else target_arch
                return f'"{script_path}" {arch}'
        if 'VS150COMNTOOLS' in os.environ and has_arch_values:
            script_path = os.environ['VS150COMNTOOLS'] + 'VsDevCmd.bat'
            if os.path.exists(script_path):
                return '"%s" -arch=%s -host_arch=%s' % (script_path, os.environ['VSCMD_ARG_TGT_ARCH'], os.environ['VSCMD_ARG_HOST_ARCH'])
        return ''

    def get_obj_target_deps(self, obj_list):
        result = {}
        for o in obj_list:
            if isinstance(o, build.ExtractedObjects):
                result[o.target.get_id()] = o.target
        return result.items()

    def get_target_deps(self, t: T.Dict[T.Any, build.Target], recursive=False):
        all_deps: T.Dict[str, build.Target] = {}
        for target in t.values():
            if isinstance(target, build.CustomTarget):
                for d in target.get_target_dependencies():
                    if isinstance(d, build.Target):
                        all_deps[d.get_id()] = d
            elif isinstance(target, build.RunTarget):
                for d in target.get_dependencies():
                    all_deps[d.get_id()] = d
            elif isinstance(target, build.BuildTarget):
                for ldep in target.link_targets:
                    if isinstance(ldep, build.CustomTargetIndex):
                        all_deps[ldep.get_id()] = ldep.target
                    else:
                        all_deps[ldep.get_id()] = ldep
                for ldep in target.link_whole_targets:
                    if isinstance(ldep, build.CustomTargetIndex):
                        all_deps[ldep.get_id()] = ldep.target
                    else:
                        all_deps[ldep.get_id()] = ldep
                for ldep in target.link_depends:
                    if isinstance(ldep, build.CustomTargetIndex):
                        all_deps[ldep.get_id()] = ldep.target
                    elif isinstance(ldep, File):
                        pass
                    else:
                        all_deps[ldep.get_id()] = ldep
                for obj_id, objdep in self.get_obj_target_deps(target.objects):
                    all_deps[obj_id] = objdep
            else:
                raise MesonException(f'Unknown target type for target {target}')
            for gendep in target.get_generated_sources():
                if isinstance(gendep, build.CustomTarget):
                    all_deps[gendep.get_id()] = gendep
                elif isinstance(gendep, build.CustomTargetIndex):
                    all_deps[gendep.target.get_id()] = gendep.target
                else:
                    generator = gendep.get_generator()
                    gen_exe = generator.get_exe()
                    if isinstance(gen_exe, build.Executable):
                        all_deps[gen_exe.get_id()] = gen_exe
                    for d in itertools.chain(generator.depends, gendep.depends):
                        if isinstance(d, build.CustomTargetIndex):
                            all_deps[d.get_id()] = d.target
                        elif isinstance(d, build.Target):
                            all_deps[d.get_id()] = d
        if not t or not recursive:
            return all_deps
        ret = self.get_target_deps(all_deps, recursive)
        ret.update(all_deps)
        return ret

    def generate_solution_dirs(self, ofile: str, parents: T.Sequence[Path]) -> None:
        prj_templ = 'Project("{%s}") = "%s", "%s", "{%s}"\n'
        iterpaths = reversed(parents)
        next(iterpaths)
        for path in iterpaths:
            if path not in self.subdirs:
                basename = path.name
                identifier = generate_guid_from_path(path, 'subdir')
                parent_dir = path.parent
                parent_identifier = self.subdirs[parent_dir][0] if parent_dir != PurePath('.') else None
                self.subdirs[path] = (identifier, parent_identifier)
                prj_line = prj_templ % (self.environment.coredata.lang_guids['directory'], basename, basename, self.subdirs[path][0])
                ofile.write(prj_line)
                ofile.write('EndProject\n')

    def generate_solution(self, sln_filename: str, projlist: T.List[Project]) -> None:
        default_projlist = self.get_build_by_default_targets()
        default_projlist.update(self.get_testlike_targets())
        sln_filename_tmp = sln_filename + '~'
        with open(sln_filename_tmp, 'w', encoding='utf-8-sig') as ofile:
            ofile.write('\nMicrosoft Visual Studio Solution File, Format Version %s\n' % self.sln_file_version)
            ofile.write('# Visual Studio %s\n' % self.sln_version_comment)
            prj_templ = 'Project("{%s}") = "%s", "%s", "{%s}"\n'
            for prj in projlist:
                if self.environment.coredata.get_option(OptionKey('layout')) == 'mirror':
                    self.generate_solution_dirs(ofile, prj[1].parents)
                target = self.build.targets[prj[0]]
                lang = 'default'
                if hasattr(target, 'compilers') and target.compilers:
                    for lang_out in target.compilers.keys():
                        lang = lang_out
                        break
                prj_line = prj_templ % (self.environment.coredata.lang_guids[lang], prj[0], prj[1], prj[2])
                ofile.write(prj_line)
                target_dict = {target.get_id(): target}
                recursive_deps = self.get_target_deps(target_dict, recursive=True)
                ofile.write('EndProject\n')
                for dep, target in recursive_deps.items():
                    if prj[0] in default_projlist:
                        default_projlist[dep] = target
            test_line = prj_templ % (self.environment.coredata.lang_guids['default'], 'RUN_TESTS', 'RUN_TESTS.vcxproj', self.environment.coredata.test_guid)
            ofile.write(test_line)
            ofile.write('EndProject\n')
            if self.gen_lite:
                regen_proj_name = 'RECONFIGURE'
                regen_proj_fname = 'RECONFIGURE.vcxproj'
            else:
                regen_proj_name = 'REGEN'
                regen_proj_fname = 'REGEN.vcxproj'
            regen_line = prj_templ % (self.environment.coredata.lang_guids['default'], regen_proj_name, regen_proj_fname, self.environment.coredata.regen_guid)
            ofile.write(regen_line)
            ofile.write('EndProject\n')
            install_line = prj_templ % (self.environment.coredata.lang_guids['default'], 'RUN_INSTALL', 'RUN_INSTALL.vcxproj', self.environment.coredata.install_guid)
            ofile.write(install_line)
            ofile.write('EndProject\n')
            ofile.write('Global\n')
            ofile.write('\tGlobalSection(SolutionConfigurationPlatforms) = preSolution\n')
            multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list() if self.gen_lite else [self.buildtype]
            for buildtype in multi_config_buildtype_list:
                ofile.write('\t\t%s|%s = %s|%s\n' % (buildtype, self.platform, buildtype, self.platform))
            ofile.write('\tEndGlobalSection\n')
            ofile.write('\tGlobalSection(ProjectConfigurationPlatforms) = postSolution\n')
            for buildtype in multi_config_buildtype_list:
                ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (self.environment.coredata.regen_guid, buildtype, self.platform, buildtype, self.platform))
                if not self.gen_lite:
                    ofile.write('\t\t{%s}.%s|%s.Build.0 = %s|%s\n' % (self.environment.coredata.regen_guid, buildtype, self.platform, buildtype, self.platform))
            for project_index, p in enumerate(projlist):
                if p[3] is MachineChoice.BUILD:
                    config_platform = self.build_platform
                else:
                    config_platform = self.platform
                for buildtype in multi_config_buildtype_list:
                    ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (p[2], buildtype, self.platform, buildtype, config_platform))
                    if (not self.gen_lite or project_index == 0) and p[0] in default_projlist and (not isinstance(self.build.targets[p[0]], build.RunTarget)):
                        ofile.write('\t\t{%s}.%s|%s.Build.0 = %s|%s\n' % (p[2], buildtype, self.platform, buildtype, config_platform))
            for buildtype in multi_config_buildtype_list:
                ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (self.environment.coredata.test_guid, buildtype, self.platform, buildtype, self.platform))
                ofile.write('\t\t{%s}.%s|%s.ActiveCfg = %s|%s\n' % (self.environment.coredata.install_guid, buildtype, self.platform, buildtype, self.platform))
            ofile.write('\tEndGlobalSection\n')
            ofile.write('\tGlobalSection(SolutionProperties) = preSolution\n')
            ofile.write('\t\tHideSolutionNode = FALSE\n')
            ofile.write('\tEndGlobalSection\n')
            if self.subdirs:
                ofile.write('\tGlobalSection(NestedProjects) = preSolution\n')
                for p in projlist:
                    if p[1].parent != PurePath('.'):
                        ofile.write('\t\t{{{}}} = {{{}}}\n'.format(p[2], self.subdirs[p[1].parent][0]))
                for subdir in self.subdirs.values():
                    if subdir[1]:
                        ofile.write('\t\t{{{}}} = {{{}}}\n'.format(subdir[0], subdir[1]))
                ofile.write('\tEndGlobalSection\n')
            ofile.write('EndGlobal\n')
        replace_if_different(sln_filename, sln_filename_tmp)

    def generate_projects(self, vslite_ctx: dict=None) -> T.List[Project]:
        startup_project = self.environment.coredata.options[OptionKey('backend_startup_project')].value
        projlist: T.List[Project] = []
        startup_idx = 0
        for i, (name, target) in enumerate(self.build.targets.items()):
            if startup_project and startup_project == target.get_basename():
                startup_idx = i
            outdir = Path(self.environment.get_build_dir(), self.get_target_dir(target))
            outdir.mkdir(exist_ok=True, parents=True)
            fname = name + '.vcxproj'
            target_dir = PurePath(self.get_target_dir(target))
            relname = target_dir / fname
            projfile_path = outdir / fname
            proj_uuid = self.environment.coredata.target_guids[name]
            generated = self.gen_vcxproj(target, str(projfile_path), proj_uuid, vslite_ctx)
            if generated:
                projlist.append((name, relname, proj_uuid, target.for_machine))
        if startup_idx:
            projlist.insert(0, projlist.pop(startup_idx))
        return projlist

    def split_sources(self, srclist):
        sources = []
        headers = []
        objects = []
        languages = []
        for i in srclist:
            if self.environment.is_header(i):
                headers.append(i)
            elif self.environment.is_object(i):
                objects.append(i)
            elif self.environment.is_source(i):
                sources.append(i)
                lang = self.lang_from_source_file(i)
                if lang not in languages:
                    languages.append(lang)
            elif self.environment.is_library(i):
                pass
            else:
                headers.append(i)
        return (sources, headers, objects, languages)

    def target_to_build_root(self, target):
        if self.get_target_dir(target) == '':
            return ''
        directories = os.path.normpath(self.get_target_dir(target)).split(os.sep)
        return os.sep.join(['..'] * len(directories))

    def quote_arguments(self, arr):
        return ['"%s"' % i for i in arr]

    def add_project_reference(self, root: ET.Element, include: str, projid: str, link_outputs: bool=False) -> None:
        ig = ET.SubElement(root, 'ItemGroup')
        pref = ET.SubElement(ig, 'ProjectReference', Include=include)
        ET.SubElement(pref, 'Project').text = '{%s}' % projid
        if not link_outputs:
            ET.SubElement(pref, 'LinkLibraryDependencies').text = 'false'

    def add_target_deps(self, root: ET.Element, target):
        target_dict = {target.get_id(): target}
        for dep in self.get_target_deps(target_dict).values():
            if dep.get_id() in self.handled_target_deps[target.get_id()]:
                continue
            relpath = self.get_target_dir_relative_to(dep, target)
            vcxproj = os.path.join(relpath, dep.get_id() + '.vcxproj')
            tid = self.environment.coredata.target_guids[dep.get_id()]
            self.add_project_reference(root, vcxproj, tid)

    def create_basic_project(self, target_name, *, temp_dir, guid, conftype='Utility', target_ext=None, target_platform=None) -> T.Tuple[ET.Element, ET.Element]:
        root = ET.Element('Project', {'DefaultTargets': 'Build', 'ToolsVersion': '4.0', 'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003'})
        confitems = ET.SubElement(root, 'ItemGroup', {'Label': 'ProjectConfigurations'})
        if not target_platform:
            target_platform = self.platform
        multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list() if self.gen_lite else [self.buildtype]
        for buildtype in multi_config_buildtype_list:
            prjconf = ET.SubElement(confitems, 'ProjectConfiguration', {'Include': buildtype + '|' + target_platform})
            ET.SubElement(prjconf, 'Configuration').text = buildtype
            ET.SubElement(prjconf, 'Platform').text = target_platform
        globalgroup = ET.SubElement(root, 'PropertyGroup', Label='Globals')
        guidelem = ET.SubElement(globalgroup, 'ProjectGuid')
        guidelem.text = '{%s}' % guid
        kw = ET.SubElement(globalgroup, 'Keyword')
        kw.text = self.platform + 'Proj'
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.Default.props')
        type_config = ET.SubElement(root, 'PropertyGroup', Label='Configuration')
        ET.SubElement(type_config, 'ConfigurationType').text = conftype
        if self.platform_toolset:
            ET.SubElement(type_config, 'PlatformToolset').text = self.platform_toolset
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.props')
        pname = ET.SubElement(globalgroup, 'ProjectName')
        pname.text = target_name
        if not self.gen_lite:
            ns = ET.SubElement(globalgroup, 'RootNamespace')
            ns.text = target_name
            p = ET.SubElement(globalgroup, 'Platform')
            p.text = target_platform
            if self.windows_target_platform_version:
                ET.SubElement(globalgroup, 'WindowsTargetPlatformVersion').text = self.windows_target_platform_version
            ET.SubElement(globalgroup, 'UseMultiToolTask').text = 'true'
            ET.SubElement(type_config, 'CharacterSet').text = 'MultiByte'
            ET.SubElement(type_config, 'UseOfMfc').text = 'false'
            direlem = ET.SubElement(root, 'PropertyGroup')
            fver = ET.SubElement(direlem, '_ProjectFileVersion')
            fver.text = self.project_file_version
            outdir = ET.SubElement(direlem, 'OutDir')
            outdir.text = '.\\'
            intdir = ET.SubElement(direlem, 'IntDir')
            intdir.text = temp_dir + '\\'
            tname = ET.SubElement(direlem, 'TargetName')
            tname.text = target_name
            if target_ext:
                ET.SubElement(direlem, 'TargetExt').text = target_ext
            ET.SubElement(direlem, 'EmbedManifest').text = 'false'
        return (root, type_config)

    def gen_run_target_vcxproj(self, target: build.RunTarget, ofname: str, guid: str) -> None:
        root, type_config = self.create_basic_project(target.name, temp_dir=target.get_id(), guid=guid)
        depend_files = self.get_target_depend_files(target)
        if not target.command:
            assert isinstance(target, build.AliasTarget)
            assert len(depend_files) == 0
        else:
            assert not isinstance(target, build.AliasTarget)
            target_env = self.get_run_target_env(target)
            _, _, cmd_raw = self.eval_custom_target_command(target)
            wrapper_cmd, _ = self.as_meson_exe_cmdline(target.command[0], cmd_raw[1:], force_serialize=True, env=target_env, verbose=True)
            self.add_custom_build(root, 'run_target', ' '.join(self.quote_arguments(wrapper_cmd)), deps=depend_files)
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
        self.add_regen_dependency(root)
        self.add_target_deps(root, target)
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)

    def gen_custom_target_vcxproj(self, target: build.CustomTarget, ofname: str, guid: str) -> None:
        if target.for_machine is MachineChoice.BUILD:
            platform = self.build_platform
        else:
            platform = self.platform
        root, type_config = self.create_basic_project(target.name, temp_dir=target.get_id(), guid=guid, target_platform=platform)
        target.absolute_paths = True
        srcs, ofilenames, cmd = self.eval_custom_target_command(target, True)
        depend_files = self.get_target_depend_files(target, True)
        tdir_abs = os.path.join(self.environment.get_build_dir(), self.get_target_dir(target))
        extra_bdeps = target.get_transitive_build_target_deps()
        wrapper_cmd, _ = self.as_meson_exe_cmdline(target.command[0], cmd[1:], workdir=tdir_abs, extra_bdeps=extra_bdeps, capture=ofilenames[0] if target.capture else None, feed=srcs[0] if target.feed else None, force_serialize=True, env=target.env, verbose=target.console)
        if target.build_always_stale:
            ofilenames += [self.nonexistent_file(os.path.join(self.environment.get_scratch_dir(), 'outofdate.file'))]
        self.add_custom_build(root, 'custom_target', ' '.join(self.quote_arguments(wrapper_cmd)), deps=wrapper_cmd[-1:] + srcs + depend_files, outputs=ofilenames, verify_files=not target.build_always_stale)
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
        self.generate_custom_generator_commands(target, root)
        self.add_regen_dependency(root)
        self.add_target_deps(root, target)
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)

    def gen_compile_target_vcxproj(self, target: build.CompileTarget, ofname: str, guid: str) -> None:
        if target.for_machine is MachineChoice.BUILD:
            platform = self.build_platform
        else:
            platform = self.platform
        root, type_config = self.create_basic_project(target.name, temp_dir=target.get_id(), guid=guid, target_platform=platform)
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
        target.generated = [self.compile_target_to_generator(target)]
        target.sources = []
        self.generate_custom_generator_commands(target, root)
        self.add_regen_dependency(root)
        self.add_target_deps(root, target)
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)

    @classmethod
    def lang_from_source_file(cls, src):
        ext = src.split('.')[-1]
        if ext in compilers.c_suffixes:
            return 'c'
        if ext in compilers.cpp_suffixes:
            return 'cpp'
        raise MesonException(f'Could not guess language from source file {src}.')

    def add_pch(self, pch_sources, lang, inc_cl):
        if lang in pch_sources:
            self.use_pch(pch_sources, lang, inc_cl)

    def create_pch(self, pch_sources, lang, inc_cl):
        pch = ET.SubElement(inc_cl, 'PrecompiledHeader')
        pch.text = 'Create'
        self.add_pch_files(pch_sources, lang, inc_cl)

    def use_pch(self, pch_sources, lang, inc_cl):
        pch = ET.SubElement(inc_cl, 'PrecompiledHeader')
        pch.text = 'Use'
        header = self.add_pch_files(pch_sources, lang, inc_cl)
        pch_include = ET.SubElement(inc_cl, 'ForcedIncludeFiles')
        pch_include.text = header + ';%(ForcedIncludeFiles)'

    def add_pch_files(self, pch_sources, lang, inc_cl):
        header = os.path.basename(pch_sources[lang][0])
        pch_file = ET.SubElement(inc_cl, 'PrecompiledHeaderFile')
        pch_file.text = header
        pch_out = ET.SubElement(inc_cl, 'PrecompiledHeaderOutputFile')
        pch_out.text = f'$(IntDir)$(TargetName)-{lang}.pch'
        pch_pdb = ET.SubElement(inc_cl, 'ProgramDataBaseFileName')
        pch_pdb.text = f'$(IntDir)$(TargetName)-{lang}.pdb'
        return header

    def is_argument_with_msbuild_xml_entry(self, entry):
        if entry[1:].startswith('fsanitize'):
            return True
        return entry[1:].startswith('M')

    def add_additional_options(self, lang, parent_node, file_args):
        args = []
        for arg in file_args[lang].to_native():
            if self.is_argument_with_msbuild_xml_entry(arg):
                continue
            if arg == '%(AdditionalOptions)':
                args.append(arg)
            else:
                args.append(self.escape_additional_option(arg))
        ET.SubElement(parent_node, 'AdditionalOptions').text = ' '.join(args)

    def add_project_nmake_defs_incs_and_opts(self, parent_node, src: str, defs_paths_opts_per_lang_and_buildtype: dict, platform: str):
        ext = src.split('.')[-1]
        lang = compilers.compilers.SUFFIX_TO_LANG.get(ext, None)
        if lang in defs_paths_opts_per_lang_and_buildtype.keys():
            for buildtype in coredata.get_genvs_default_buildtype_list():
                defs, paths, opts = defs_paths_opts_per_lang_and_buildtype[lang][buildtype]
                condition = f"'$(Configuration)|$(Platform)'=='{buildtype}|{platform}'"
                ET.SubElement(parent_node, 'PreprocessorDefinitions', Condition=condition).text = defs
                ET.SubElement(parent_node, 'AdditionalIncludeDirectories', Condition=condition).text = paths
                ET.SubElement(parent_node, 'AdditionalOptions', Condition=condition).text = opts
        else:
            ET.SubElement(parent_node, 'PreprocessorDefinitions').text = '$(NMakePreprocessorDefinitions)'
            ET.SubElement(parent_node, 'AdditionalIncludeDirectories').text = '$(NMakeIncludeSearchPath)'
            ET.SubElement(parent_node, 'AdditionalOptions').text = '$(AdditionalOptions)'

    def add_preprocessor_defines(self, lang, parent_node, file_defines):
        defines = []
        for define in file_defines[lang]:
            if define == '%(PreprocessorDefinitions)':
                defines.append(define)
            else:
                defines.append(self.escape_preprocessor_define(define))
        ET.SubElement(parent_node, 'PreprocessorDefinitions').text = ';'.join(defines)

    def add_include_dirs(self, lang, parent_node, file_inc_dirs):
        dirs = file_inc_dirs[lang]
        ET.SubElement(parent_node, 'AdditionalIncludeDirectories').text = ';'.join(dirs)

    @staticmethod
    def escape_preprocessor_define(define: str) -> str:
        table = str.maketrans({'%': '%25', '$': '%24', '@': '%40', "'": '%27', ';': '%3B', '?': '%3F', '*': '%2A', '\\': '\\\\'})
        return define.translate(table)

    @staticmethod
    def escape_additional_option(option: str) -> str:
        table = str.maketrans({'%': '%25', '$': '%24', '@': '%40', "'": '%27', ';': '%3B', '?': '%3F', '*': '%2A', ' ': '%20'})
        option = option.translate(table)
        if option.endswith('\\'):
            option += '\\'
        return f'"{option}"'

    @staticmethod
    def split_link_args(args):
        """
        Split a list of link arguments into three lists:
        * library search paths
        * library filenames (or paths)
        * other link arguments
        """
        lpaths = []
        libs = []
        other = []
        for arg in args:
            if arg.startswith('/LIBPATH:'):
                lpath = arg[9:]
                if lpath in lpaths:
                    lpaths.remove(lpath)
                lpaths.append(lpath)
            elif arg.startswith(('/', '-')):
                other.append(arg)
            elif arg.endswith('.lib') or arg.endswith('.a'):
                if arg not in libs:
                    libs.append(arg)
            else:
                other.append(arg)
        return (lpaths, libs, other)

    def _get_cl_compiler(self, target):
        for lang, c in target.compilers.items():
            if lang in {'c', 'cpp'}:
                return c
        if len(target.objects) > 0:
            for lang, c in self.environment.coredata.compilers[target.for_machine].items():
                if lang in {'c', 'cpp'}:
                    return c
        raise MesonException('Could not find a C or C++ compiler. MSVC can only build C/C++ projects.')

    def _prettyprint_vcxproj_xml(self, tree: ET.ElementTree, ofname: str) -> None:
        ofname_tmp = ofname + '~'
        tree.write(ofname_tmp, encoding='utf-8', xml_declaration=True)
        doc = xml.dom.minidom.parse(ofname_tmp)
        with open(ofname_tmp, 'w', encoding='utf-8') as of:
            of.write(doc.toprettyxml())
        replace_if_different(ofname, ofname_tmp)

    def get_args_defines_and_inc_dirs(self, target, compiler, generated_files_include_dirs, proj_to_src_root, proj_to_src_dir, build_args):
        target_args = []
        target_defines = []
        target_inc_dirs = []
        file_args: T.Dict[str, CompilerArgs] = {l: c.compiler_args() for l, c in target.compilers.items()}
        file_defines = {l: [] for l in target.compilers}
        file_inc_dirs = {l: [] for l in target.compilers}
        for l, comp in target.compilers.items():
            if l in file_args:
                file_args[l] += compilers.get_base_compile_args(target.get_options(), comp)
                file_args[l] += comp.get_option_compile_args(target.get_options())
        for l, args in self.build.projects_args[target.for_machine].get(target.subproject, {}).items():
            if l in file_args:
                file_args[l] += args
        for l, args in self.build.global_args[target.for_machine].items():
            if l in file_args:
                file_args[l] += args
        for l in file_args.keys():
            file_args[l] += target.get_option(OptionKey('args', machine=target.for_machine, lang=l))
        for args in file_args.values():
            args += ['%(AdditionalOptions)', '%(PreprocessorDefinitions)', '%(AdditionalIncludeDirectories)']
            args += ['-I' + arg for arg in generated_files_include_dirs]
            for d in reversed(target.get_include_dirs()):
                for i in reversed(d.get_incdirs()):
                    curdir = os.path.join(d.get_curdir(), i)
                    try:
                        args.append('-I' + os.path.join(proj_to_src_root, curdir))
                        args.append('-I' + self.relpath(curdir, target.subdir))
                    except ValueError:
                        args.append('-I' + os.path.normpath(curdir))
                for i in d.get_extra_build_dirs():
                    curdir = os.path.join(d.get_curdir(), i)
                    args.append('-I' + self.relpath(curdir, target.subdir))
        for l, args in target.extra_args.items():
            if l in file_args:
                file_args[l] += args
        for args in file_args.values():
            t_inc_dirs = [self.relpath(self.get_target_private_dir(target), self.get_target_dir(target))]
            if target.implicit_include_directories:
                t_inc_dirs += ['.', proj_to_src_dir]
            args += ['-I' + arg for arg in t_inc_dirs]
        for l, args in file_args.items():
            for arg in args[:]:
                if arg.startswith(('-D', '/D')) or arg == '%(PreprocessorDefinitions)':
                    file_args[l].remove(arg)
                    if arg == '%(PreprocessorDefinitions)':
                        define = arg
                    else:
                        define = arg[2:]
                    if define not in file_defines[l]:
                        file_defines[l].append(define)
                elif arg.startswith(('-I', '/I')) or arg == '%(AdditionalIncludeDirectories)':
                    file_args[l].remove(arg)
                    if arg == '%(AdditionalIncludeDirectories)':
                        inc_dir = arg
                    else:
                        inc_dir = arg[2:]
                    if inc_dir not in file_inc_dirs[l]:
                        file_inc_dirs[l].append(inc_dir)
                    if inc_dir not in target_inc_dirs:
                        target_inc_dirs.append(inc_dir)
        for d in reversed(target.get_external_deps()):
            if d.name != 'openmp':
                d_compile_args = compiler.unix_args_to_native(d.get_compile_args())
                for arg in d_compile_args:
                    if arg.startswith(('-D', '/D')):
                        define = arg[2:]
                        if define in target_defines:
                            target_defines.remove(define)
                        target_defines.append(define)
                    elif arg.startswith(('-I', '/I')):
                        inc_dir = arg[2:]
                        if inc_dir not in target_inc_dirs:
                            target_inc_dirs.append(inc_dir)
                    else:
                        target_args.append(arg)
        if '/Gw' in build_args:
            target_args.append('/Gw')
        return ((target_args, file_args), (target_defines, file_defines), (target_inc_dirs, file_inc_dirs))

    @staticmethod
    def get_build_args(compiler, optimization_level: str, debug: bool, sanitize: str) -> T.List[str]:
        build_args = compiler.get_optimization_args(optimization_level)
        build_args += compiler.get_debug_args(debug)
        build_args += compiler.sanitizer_compile_args(sanitize)
        return build_args

    @staticmethod
    def _extract_nmake_fields(captured_build_args: list[str]) -> T.Tuple[str, str, str]:
        include_dir_options = ['-I', '/I', '-isystem', '/clang:-isystem', '/imsvc', '/external:I']
        defs = ''
        paths = '$(VC_IncludePath);$(WindowsSDK_IncludePath);'
        additional_opts = ''
        for arg in captured_build_args:
            if arg.startswith(('-D', '/D')):
                defs += arg[2:] + ';'
            else:
                opt_match = next((opt for opt in include_dir_options if arg.startswith(opt)), None)
                if opt_match:
                    paths += arg[len(opt_match):] + ';'
                elif arg.startswith(('-', '/')):
                    additional_opts += arg + ' '
        return (defs, paths, additional_opts)

    @staticmethod
    def get_nmake_base_meson_command_and_exe_search_paths() -> T.Tuple[str, str]:
        meson_cmd_list = mesonlib.get_meson_command()
        assert len(meson_cmd_list) == 1 or len(meson_cmd_list) == 2
        exe_search_paths = os.path.dirname(meson_cmd_list[0])
        nmake_base_meson_command = os.path.basename(meson_cmd_list[0])
        if len(meson_cmd_list) != 1:
            nmake_base_meson_command += ' "' + meson_cmd_list[1] + '"'
            exe_search_paths += ';' + os.path.dirname(meson_cmd_list[1])
        exe_search_paths += ';C:\\Windows\\system32;C:\\Windows'
        return (nmake_base_meson_command, exe_search_paths)

    def add_gen_lite_makefile_vcxproj_elements(self, root: ET.Element, platform: str, target_ext: str, vslite_ctx: dict, target, proj_to_build_root: str, primary_src_lang: T.Optional[str]) -> None:
        ET.SubElement(root, 'ImportGroup', Label='ExtensionSettings')
        ET.SubElement(root, 'ImportGroup', Label='Shared')
        prop_sheets_grp = ET.SubElement(root, 'ImportGroup', Label='PropertySheets')
        ET.SubElement(prop_sheets_grp, 'Import', {'Project': '$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props', 'Condition': "exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')", 'Label': 'LocalAppDataPlatform'})
        ET.SubElement(root, 'PropertyGroup', Label='UserMacros')
        nmake_base_meson_command, exe_search_paths = Vs2010Backend.get_nmake_base_meson_command_and_exe_search_paths()
        proj_to_multiconfigured_builds_parent_dir = os.path.join(proj_to_build_root, '..')
        multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list()
        for buildtype in multi_config_buildtype_list:
            per_config_prop_group = ET.SubElement(root, 'PropertyGroup', Condition=f"'$(Configuration)|$(Platform)'=='{buildtype}|{platform}'")
            _, build_dir_tail = os.path.split(self.src_to_build)
            meson_build_dir_for_buildtype = build_dir_tail[:-2] + buildtype
            proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
            ET.SubElement(per_config_prop_group, 'OutDir').text = f'{proj_to_build_dir_for_buildtype}\\'
            ET.SubElement(per_config_prop_group, 'IntDir').text = f'{proj_to_build_dir_for_buildtype}\\'
            ET.SubElement(per_config_prop_group, 'NMakeBuildCommandLine').text = f'{nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}"'
            ET.SubElement(per_config_prop_group, 'NMakeOutput').text = f'$(OutDir){target.name}{target_ext}'
            captured_build_args = vslite_ctx[buildtype][target.get_id()]
            ET.SubElement(per_config_prop_group, 'NMakeReBuildCommandLine').text = f'{nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}" --clean && {nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}"'
            ET.SubElement(per_config_prop_group, 'NMakeCleanCommandLine').text = f'{nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}" --clean'
            ET.SubElement(per_config_prop_group, 'ExecutablePath').text = exe_search_paths
            if primary_src_lang:
                primary_src_type_build_args = captured_build_args[primary_src_lang]
                preproc_defs, inc_paths, other_compile_opts = Vs2010Backend._extract_nmake_fields(primary_src_type_build_args)
                ET.SubElement(per_config_prop_group, 'NMakePreprocessorDefinitions').text = preproc_defs
                ET.SubElement(per_config_prop_group, 'NMakeIncludeSearchPath').text = inc_paths
                ET.SubElement(per_config_prop_group, 'AdditionalOptions').text = other_compile_opts
            ET.SubElement(per_config_prop_group, 'IncludePath')
            ET.SubElement(per_config_prop_group, 'ExternalIncludePath')
            ET.SubElement(per_config_prop_group, 'ReferencePath')
            ET.SubElement(per_config_prop_group, 'LibraryPath')
            ET.SubElement(per_config_prop_group, 'LibraryWPath')
            ET.SubElement(per_config_prop_group, 'SourcePath')
            ET.SubElement(per_config_prop_group, 'ExcludePath')

    def add_non_makefile_vcxproj_elements(self, root: ET.Element, type_config: ET.Element, target, platform: str, subsystem, build_args, target_args, target_defines, target_inc_dirs, file_args) -> None:
        compiler = self._get_cl_compiler(target)
        buildtype_link_args = compiler.get_optimization_link_args(self.optimization)
        down = self.target_to_build_root(target)
        ET.SubElement(type_config, 'WholeProgramOptimization').text = 'false'
        ET.SubElement(type_config, 'BasicRuntimeChecks').text = 'Default'
        if '/INCREMENTAL:NO' in buildtype_link_args:
            ET.SubElement(type_config, 'LinkIncremental').text = 'false'
        compiles = ET.SubElement(root, 'ItemDefinitionGroup')
        clconf = ET.SubElement(compiles, 'ClCompile')
        if True in (dep.name == 'openmp' for dep in target.get_external_deps()):
            ET.SubElement(clconf, 'OpenMPSupport').text = 'true'
        vscrt_type = target.get_option(OptionKey('b_vscrt'))
        vscrt_val = compiler.get_crt_val(vscrt_type, self.buildtype)
        if vscrt_val == 'mdd':
            ET.SubElement(type_config, 'UseDebugLibraries').text = 'true'
            ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreadedDebugDLL'
        elif vscrt_val == 'mt':
            ET.SubElement(type_config, 'UseDebugLibraries').text = 'false'
            ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreaded'
        elif vscrt_val == 'mtd':
            ET.SubElement(type_config, 'UseDebugLibraries').text = 'true'
            ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreadedDebug'
        else:
            ET.SubElement(type_config, 'UseDebugLibraries').text = 'false'
            ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreadedDLL'
        if '/fsanitize=address' in build_args:
            ET.SubElement(type_config, 'EnableASAN').text = 'true'
        if '/ZI' in build_args:
            ET.SubElement(clconf, 'DebugInformationFormat').text = 'EditAndContinue'
        elif '/Zi' in build_args:
            ET.SubElement(clconf, 'DebugInformationFormat').text = 'ProgramDatabase'
        elif '/Z7' in build_args:
            ET.SubElement(clconf, 'DebugInformationFormat').text = 'OldStyle'
        else:
            ET.SubElement(clconf, 'DebugInformationFormat').text = 'None'
        if '/RTC1' in build_args:
            ET.SubElement(clconf, 'BasicRuntimeChecks').text = 'EnableFastChecks'
        elif '/RTCu' in build_args:
            ET.SubElement(clconf, 'BasicRuntimeChecks').text = 'UninitializedLocalUsageCheck'
        elif '/RTCs' in build_args:
            ET.SubElement(clconf, 'BasicRuntimeChecks').text = 'StackFrameRuntimeCheck'
        if 'cpp' in target.compilers:
            eh = target.get_option(OptionKey('eh', machine=target.for_machine, lang='cpp'))
            if eh == 'a':
                ET.SubElement(clconf, 'ExceptionHandling').text = 'Async'
            elif eh == 's':
                ET.SubElement(clconf, 'ExceptionHandling').text = 'SyncCThrow'
            elif eh == 'none':
                ET.SubElement(clconf, 'ExceptionHandling').text = 'false'
            else:
                ET.SubElement(clconf, 'ExceptionHandling').text = 'Sync'
        if len(target_args) > 0:
            target_args.append('%(AdditionalOptions)')
            ET.SubElement(clconf, 'AdditionalOptions').text = ' '.join(target_args)
        ET.SubElement(clconf, 'AdditionalIncludeDirectories').text = ';'.join(target_inc_dirs)
        target_defines.append('%(PreprocessorDefinitions)')
        ET.SubElement(clconf, 'PreprocessorDefinitions').text = ';'.join(target_defines)
        ET.SubElement(clconf, 'FunctionLevelLinking').text = 'true'
        warning_level = T.cast('str', target.get_option(OptionKey('warning_level')))
        warning_level = 'EnableAllWarnings' if warning_level == 'everything' else 'Level' + str(1 + int(warning_level))
        ET.SubElement(clconf, 'WarningLevel').text = warning_level
        if target.get_option(OptionKey('werror')):
            ET.SubElement(clconf, 'TreatWarningAsError').text = 'true'
        o_flags = split_o_flags_args(build_args)
        if '/Ox' in o_flags:
            ET.SubElement(clconf, 'Optimization').text = 'Full'
        elif '/O2' in o_flags:
            ET.SubElement(clconf, 'Optimization').text = 'MaxSpeed'
        elif '/O1' in o_flags:
            ET.SubElement(clconf, 'Optimization').text = 'MinSpace'
        elif '/Od' in o_flags:
            ET.SubElement(clconf, 'Optimization').text = 'Disabled'
        if '/Oi' in o_flags:
            ET.SubElement(clconf, 'IntrinsicFunctions').text = 'true'
        if '/Ob1' in o_flags:
            ET.SubElement(clconf, 'InlineFunctionExpansion').text = 'OnlyExplicitInline'
        elif '/Ob2' in o_flags:
            ET.SubElement(clconf, 'InlineFunctionExpansion').text = 'AnySuitable'
        if '/Os' in o_flags or '/O1' in o_flags:
            ET.SubElement(clconf, 'FavorSizeOrSpeed').text = 'Size'
        elif '/Od' not in o_flags:
            ET.SubElement(clconf, 'FavorSizeOrSpeed').text = 'Speed'
        self.generate_lang_standard_info(file_args, clconf)
        resourcecompile = ET.SubElement(compiles, 'ResourceCompile')
        ET.SubElement(resourcecompile, 'PreprocessorDefinitions')
        link = ET.SubElement(compiles, 'Link')
        extra_link_args = compiler.compiler_args()
        extra_link_args += compiler.get_optimization_link_args(self.optimization)
        if self.debug:
            self.generate_debug_information(link)
        else:
            ET.SubElement(link, 'GenerateDebugInformation').text = 'false'
        if not isinstance(target, build.StaticLibrary):
            if isinstance(target, build.SharedModule):
                extra_link_args += compiler.get_std_shared_module_link_args(target.get_options())
            extra_link_args += self.build.get_project_link_args(compiler, target.subproject, target.for_machine)
            extra_link_args += self.build.get_global_link_args(compiler, target.for_machine)
            extra_link_args += self.environment.coredata.get_external_link_args(target.for_machine, compiler.get_language())
            extra_link_args += target.link_args
            for dep in target.get_external_deps():
                if dep.name == 'openmp':
                    ET.SubElement(clconf, 'OpenMPSupport').text = 'true'
                else:
                    extra_link_args.extend_direct(dep.get_link_args())
            for d in target.get_dependencies():
                if isinstance(d, build.StaticLibrary):
                    for dep in d.get_external_deps():
                        if dep.name == 'openmp':
                            ET.SubElement(clconf, 'OpenMPSupport').text = 'true'
                        else:
                            extra_link_args.extend_direct(dep.get_link_args())
        extra_link_args += compiler.get_option_link_args(target.get_options())
        additional_libpaths, additional_links, extra_link_args = self.split_link_args(extra_link_args.to_native())
        for t in target.get_dependencies():
            if isinstance(t, build.CustomTargetIndex):
                lobj = t
            else:
                lobj = self.build.targets[t.get_id()]
            linkname = os.path.join(down, self.get_target_filename_for_linking(lobj))
            if t in target.link_whole_targets:
                if compiler.id == 'msvc' and version_compare(compiler.version, '<19.00.23918'):
                    l = t.extract_all_objects(False)
                    for gen in l.genlist:
                        for src in gen.get_outputs():
                            if self.environment.is_source(src):
                                path = self.get_target_generated_dir(t, gen, src)
                                gen_src_ext = '.' + os.path.splitext(path)[1][1:]
                                extra_link_args.append(path[:-len(gen_src_ext)] + '.obj')
                    for src in l.srclist:
                        if self.environment.is_source(src):
                            target_private_dir = self.relpath(self.get_target_private_dir(t), self.get_target_dir(t))
                            rel_obj = self.object_filename_from_source(t, src, target_private_dir)
                            extra_link_args.append(rel_obj)
                    extra_link_args.extend(self.flatten_object_list(t))
                else:
                    extra_link_args += compiler.get_link_whole_for(linkname)
                trelpath = self.get_target_dir_relative_to(t, target)
                tvcxproj = os.path.join(trelpath, t.get_id() + '.vcxproj')
                tid = self.environment.coredata.target_guids[t.get_id()]
                self.add_project_reference(root, tvcxproj, tid, link_outputs=True)
                self.handled_target_deps[target.get_id()].append(t.get_id())
            elif linkname not in additional_links:
                additional_links.append(linkname)
        for lib in self.get_custom_target_provided_libraries(target):
            additional_links.append(self.relpath(lib, self.get_target_dir(target)))
        if len(extra_link_args) > 0:
            extra_link_args.append('%(AdditionalOptions)')
            ET.SubElement(link, 'AdditionalOptions').text = ' '.join(extra_link_args)
        if len(additional_libpaths) > 0:
            additional_libpaths.insert(0, '%(AdditionalLibraryDirectories)')
            ET.SubElement(link, 'AdditionalLibraryDirectories').text = ';'.join(additional_libpaths)
        if len(additional_links) > 0:
            additional_links.append('%(AdditionalDependencies)')
            ET.SubElement(link, 'AdditionalDependencies').text = ';'.join(additional_links)
        ofile = ET.SubElement(link, 'OutputFile')
        ofile.text = f'$(OutDir){target.get_filename()}'
        subsys = ET.SubElement(link, 'SubSystem')
        subsys.text = subsystem
        if isinstance(target, (build.SharedLibrary, build.Executable)) and target.get_import_filename():
            ET.SubElement(link, 'ImportLibrary').text = target.get_import_filename()
        if isinstance(target, (build.SharedLibrary, build.Executable)):
            if target.vs_module_defs:
                relpath = os.path.join(down, target.vs_module_defs.rel_to_builddir(self.build_to_src))
                ET.SubElement(link, 'ModuleDefinitionFile').text = relpath
        if self.debug:
            pdb = ET.SubElement(link, 'ProgramDataBaseFileName')
            pdb.text = f'$(OutDir){target.name}.pdb'
        targetmachine = ET.SubElement(link, 'TargetMachine')
        if target.for_machine is MachineChoice.BUILD:
            targetplatform = platform.lower()
        else:
            targetplatform = self.platform.lower()
        if targetplatform == 'win32':
            targetmachine.text = 'MachineX86'
        elif targetplatform == 'x64' or detect_microsoft_gdk(targetplatform):
            targetmachine.text = 'MachineX64'
        elif targetplatform == 'arm':
            targetmachine.text = 'MachineARM'
        elif targetplatform == 'arm64':
            targetmachine.text = 'MachineARM64'
        elif targetplatform == 'arm64ec':
            targetmachine.text = 'MachineARM64EC'
        else:
            raise MesonException('Unsupported Visual Studio target machine: ' + targetplatform)
        ET.SubElement(link, 'SuppressStartupBanner').text = 'true'
        if not target.get_option(OptionKey('debug')):
            ET.SubElement(link, 'SetChecksum').text = 'true'

    def relocate_generated_file_paths_to_concrete_build_dir(self, gen_files: T.List[str], target: T.Union[build.Target, build.CustomTargetIndex]) -> None:
        _, build_dir_tail = os.path.split(self.src_to_build)
        meson_build_dir_for_buildtype = build_dir_tail[:-2] + coredata.get_genvs_default_buildtype_list()[0]
        proj_to_build_root = self.target_to_build_root(target)
        proj_to_multiconfigured_builds_parent_dir = os.path.join(proj_to_build_root, '..')
        proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
        relocate_to_concrete_builddir_target = os.path.normpath(os.path.join(proj_to_build_dir_for_buildtype, self.get_target_dir(target)))
        for idx, file_path in enumerate(gen_files):
            gen_files[idx] = os.path.normpath(os.path.join(relocate_to_concrete_builddir_target, file_path))

    def gen_vcxproj(self, target: build.BuildTarget, ofname: str, guid: str, vslite_ctx: dict=None) -> bool:
        mlog.debug(f'Generating vcxproj {target.name}.')
        subsystem = 'Windows'
        self.handled_target_deps[target.get_id()] = []
        if self.gen_lite:
            if not isinstance(target, build.BuildTarget):
                return False
            conftype = 'Makefile'
        elif isinstance(target, build.Executable):
            conftype = 'Application'
            subsystem = target.win_subsystem.split(',')[0]
        elif isinstance(target, build.StaticLibrary):
            conftype = 'StaticLibrary'
        elif isinstance(target, build.SharedLibrary):
            conftype = 'DynamicLibrary'
        elif isinstance(target, build.CustomTarget):
            self.gen_custom_target_vcxproj(target, ofname, guid)
            return True
        elif isinstance(target, build.RunTarget):
            self.gen_run_target_vcxproj(target, ofname, guid)
            return True
        elif isinstance(target, build.CompileTarget):
            self.gen_compile_target_vcxproj(target, ofname, guid)
            return True
        else:
            raise MesonException(f'Unknown target type for {target.get_basename()}')
        sources, headers, objects, _languages = self.split_sources(target.sources)
        if target.is_unity:
            sources = self.generate_unity_files(target, sources)
        if target.for_machine is MachineChoice.BUILD:
            platform = self.build_platform
        else:
            platform = self.platform
        tfilename = os.path.splitext(target.get_filename())
        root, type_config = self.create_basic_project(tfilename[0], temp_dir=target.get_id(), guid=guid, conftype=conftype, target_ext=tfilename[1], target_platform=platform)
        generated_files, custom_target_output_files, generated_files_include_dirs = self.generate_custom_generator_commands(target, root)
        gen_src, gen_hdrs, gen_objs, _gen_langs = self.split_sources(generated_files)
        custom_src, custom_hdrs, custom_objs, _custom_langs = self.split_sources(custom_target_output_files)
        gen_src += custom_src
        gen_hdrs += custom_hdrs
        compiler = self._get_cl_compiler(target)
        build_args = Vs2010Backend.get_build_args(compiler, self.optimization, self.debug, self.sanitize)
        assert isinstance(target, (build.Executable, build.SharedLibrary, build.StaticLibrary, build.SharedModule)), 'for mypy'
        proj_to_build_root = self.target_to_build_root(target)
        proj_to_src_root = os.path.join(proj_to_build_root, self.build_to_src)
        proj_to_src_dir = os.path.join(proj_to_src_root, self.get_target_dir(target))
        (target_args, file_args), (target_defines, file_defines), (target_inc_dirs, file_inc_dirs) = self.get_args_defines_and_inc_dirs(target, compiler, generated_files_include_dirs, proj_to_src_root, proj_to_src_dir, build_args)
        if self.gen_lite:
            assert vslite_ctx is not None
            primary_src_lang = get_primary_source_lang(target.sources, custom_src)
            self.add_gen_lite_makefile_vcxproj_elements(root, platform, tfilename[1], vslite_ctx, target, proj_to_build_root, primary_src_lang)
        else:
            self.add_non_makefile_vcxproj_elements(root, type_config, target, platform, subsystem, build_args, target_args, target_defines, target_inc_dirs, file_args)
        meson_file_group = ET.SubElement(root, 'ItemGroup')
        ET.SubElement(meson_file_group, 'None', Include=os.path.join(proj_to_src_dir, build_filename))

        def path_normalize_add(path, lis):
            normalized = os.path.normcase(os.path.normpath(path))
            if normalized not in lis:
                lis.append(normalized)
                return True
            else:
                return False
        pch_sources = {}
        if self.target_uses_pch(target):
            for lang in ['c', 'cpp']:
                pch = target.get_pch(lang)
                if not pch:
                    continue
                if compiler.id == 'msvc':
                    if len(pch) == 1:
                        src = os.path.join(proj_to_build_root, self.create_msvc_pch_implementation(target, lang, pch[0]))
                        pch_header_dir = os.path.dirname(os.path.join(proj_to_src_dir, pch[0]))
                    else:
                        src = os.path.join(proj_to_src_dir, pch[1])
                        pch_header_dir = None
                    pch_sources[lang] = [pch[0], src, lang, pch_header_dir]
                else:
                    pch_sources[lang] = [pch[0], None, lang, None]
        previous_includes = []
        if len(headers) + len(gen_hdrs) + len(target.extra_files) + len(pch_sources) > 0:
            if self.gen_lite and gen_hdrs:
                self.relocate_generated_file_paths_to_concrete_build_dir(gen_hdrs, target)
            inc_hdrs = ET.SubElement(root, 'ItemGroup')
            for h in headers:
                relpath = os.path.join(proj_to_build_root, h.rel_to_builddir(self.build_to_src))
                if path_normalize_add(relpath, previous_includes):
                    ET.SubElement(inc_hdrs, 'CLInclude', Include=relpath)
            for h in gen_hdrs:
                if path_normalize_add(h, previous_includes):
                    ET.SubElement(inc_hdrs, 'CLInclude', Include=h)
            for h in target.extra_files:
                relpath = os.path.join(proj_to_build_root, h.rel_to_builddir(self.build_to_src))
                if path_normalize_add(relpath, previous_includes):
                    ET.SubElement(inc_hdrs, 'CLInclude', Include=relpath)
            for headers in pch_sources.values():
                path = os.path.join(proj_to_src_dir, headers[0])
                if path_normalize_add(path, previous_includes):
                    ET.SubElement(inc_hdrs, 'CLInclude', Include=path)
        previous_sources = []
        if len(sources) + len(gen_src) + len(pch_sources) > 0:
            if self.gen_lite:
                defs_paths_opts_per_lang_and_buildtype = get_non_primary_lang_intellisense_fields(vslite_ctx, target.get_id(), primary_src_lang)
                if gen_src:
                    self.relocate_generated_file_paths_to_concrete_build_dir(gen_src, target)
            inc_src = ET.SubElement(root, 'ItemGroup')
            for s in sources:
                relpath = os.path.join(proj_to_build_root, s.rel_to_builddir(self.build_to_src))
                if path_normalize_add(relpath, previous_sources):
                    inc_cl = ET.SubElement(inc_src, 'CLCompile', Include=relpath)
                    if self.gen_lite:
                        self.add_project_nmake_defs_incs_and_opts(inc_cl, relpath, defs_paths_opts_per_lang_and_buildtype, platform)
                    else:
                        lang = Vs2010Backend.lang_from_source_file(s)
                        self.add_pch(pch_sources, lang, inc_cl)
                        self.add_additional_options(lang, inc_cl, file_args)
                        self.add_preprocessor_defines(lang, inc_cl, file_defines)
                        self.add_include_dirs(lang, inc_cl, file_inc_dirs)
                        ET.SubElement(inc_cl, 'ObjectFileName').text = '$(IntDir)' + self.object_filename_from_source(target, s)
            for s in gen_src:
                if path_normalize_add(s, previous_sources):
                    inc_cl = ET.SubElement(inc_src, 'CLCompile', Include=s)
                    if self.gen_lite:
                        self.add_project_nmake_defs_incs_and_opts(inc_cl, s, defs_paths_opts_per_lang_and_buildtype, platform)
                    else:
                        lang = Vs2010Backend.lang_from_source_file(s)
                        self.add_pch(pch_sources, lang, inc_cl)
                        self.add_additional_options(lang, inc_cl, file_args)
                        self.add_preprocessor_defines(lang, inc_cl, file_defines)
                        self.add_include_dirs(lang, inc_cl, file_inc_dirs)
                        s = File.from_built_file(target.get_subdir(), s)
                        ET.SubElement(inc_cl, 'ObjectFileName').text = '$(IntDir)' + self.object_filename_from_source(target, s)
            for lang, headers in pch_sources.items():
                impl = headers[1]
                if impl and path_normalize_add(impl, previous_sources):
                    inc_cl = ET.SubElement(inc_src, 'CLCompile', Include=impl)
                    self.create_pch(pch_sources, lang, inc_cl)
                    if self.gen_lite:
                        self.add_project_nmake_defs_incs_and_opts(inc_cl, impl, defs_paths_opts_per_lang_and_buildtype, platform)
                    else:
                        self.add_additional_options(lang, inc_cl, file_args)
                        self.add_preprocessor_defines(lang, inc_cl, file_defines)
                        pch_header_dir = pch_sources[lang][3]
                        if pch_header_dir:
                            inc_dirs = copy.deepcopy(file_inc_dirs)
                            inc_dirs[lang] = [pch_header_dir] + inc_dirs[lang]
                        else:
                            inc_dirs = file_inc_dirs
                        self.add_include_dirs(lang, inc_cl, inc_dirs)
        additional_objects = []
        for o in self.flatten_object_list(target, proj_to_build_root)[0]:
            assert isinstance(o, str)
            additional_objects.append(o)
        for o in custom_objs:
            additional_objects.append(o)
        explicit_link_gen_objs = [obj for obj in gen_objs if not obj.endswith(('.obj', '.res'))]
        previous_objects = []
        if len(objects) + len(additional_objects) + len(explicit_link_gen_objs) > 0:
            inc_objs = ET.SubElement(root, 'ItemGroup')
            for s in objects:
                relpath = os.path.join(proj_to_build_root, s.rel_to_builddir(self.build_to_src))
                if path_normalize_add(relpath, previous_objects):
                    ET.SubElement(inc_objs, 'Object', Include=relpath)
            for s in additional_objects + explicit_link_gen_objs:
                if path_normalize_add(s, previous_objects):
                    ET.SubElement(inc_objs, 'Object', Include=s)
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
        self.add_regen_dependency(root)
        if not self.gen_lite:
            self.add_target_deps(root, target)
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)
        if self.environment.coredata.get_option(OptionKey('layout')) == 'mirror':
            self.gen_vcxproj_filters(target, ofname)
        return True

    def gen_vcxproj_filters(self, target, ofname):
        root = ET.Element('Project', {'ToolsVersion': '4.0', 'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003'})
        filter_folders = ET.SubElement(root, 'ItemGroup')
        filter_items = ET.SubElement(root, 'ItemGroup')
        mlog.debug(f'Generating vcxproj filters {target.name}.')

        def relative_to_defined_in(file):
            return os.path.dirname(self.relpath(PureWindowsPath(file.subdir, file.fname), self.get_target_dir(target)))
        found_folders_to_filter = {}
        all_files = target.sources + target.extra_files
        for i in all_files:
            if not os.path.isabs(i.fname):
                dirname = relative_to_defined_in(i)
                if dirname:
                    found_folders_to_filter[dirname] = ''
        for folder in found_folders_to_filter:
            dirname = folder
            filter = ''
            while dirname:
                basename = os.path.basename(dirname)
                if filter == '':
                    filter = basename
                else:
                    filter = basename + ('\\' if dirname in found_folders_to_filter else '/') + filter
                dirname = os.path.dirname(dirname)
            if filter != '':
                found_folders_to_filter[folder] = filter
                filter_element = ET.SubElement(filter_folders, 'Filter', {'Include': filter})
                uuid_element = ET.SubElement(filter_element, 'UniqueIdentifier')
                uuid_element.text = '{' + str(uuid.uuid4()).upper() + '}'
        sources, headers, objects, _ = self.split_sources(all_files)
        down = self.target_to_build_root(target)

        def add_element(type_name, elements):
            for i in elements:
                if not os.path.isabs(i.fname):
                    dirname = relative_to_defined_in(i)
                    if dirname and dirname in found_folders_to_filter:
                        relpath = os.path.join(down, i.rel_to_builddir(self.build_to_src))
                        target_element = ET.SubElement(filter_items, type_name, {'Include': relpath})
                        filter_element = ET.SubElement(target_element, 'Filter')
                        filter_element.text = found_folders_to_filter[dirname]
        add_element('ClCompile', sources)
        add_element('ClInclude', headers)
        add_element('Object', objects)
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname + '.filters')

    def gen_regenproj(self):
        if self.gen_lite:
            project_name = 'RECONFIGURE'
            ofname = os.path.join(self.environment.get_build_dir(), 'RECONFIGURE.vcxproj')
            conftype = 'Makefile'
        else:
            project_name = 'REGEN'
            ofname = os.path.join(self.environment.get_build_dir(), 'REGEN.vcxproj')
            conftype = 'Utility'
        guid = self.environment.coredata.regen_guid
        root, type_config = self.create_basic_project(project_name, temp_dir='regen-temp', guid=guid, conftype=conftype)
        if self.gen_lite:
            nmake_base_meson_command, exe_search_paths = Vs2010Backend.get_nmake_base_meson_command_and_exe_search_paths()
            all_configs_prop_group = ET.SubElement(root, 'PropertyGroup')
            multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list()
            _, build_dir_tail = os.path.split(self.src_to_build)
            proj_to_multiconfigured_builds_parent_dir = '..'
            proj_to_src_dir = self.build_to_src
            reconfigure_all_cmd = ''
            for buildtype in multi_config_buildtype_list:
                meson_build_dir_for_buildtype = build_dir_tail[:-2] + buildtype
                proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
                reconfigure_all_cmd += f'{nmake_base_meson_command} setup --reconfigure "{proj_to_build_dir_for_buildtype}" "{proj_to_src_dir}"\n'
            ET.SubElement(all_configs_prop_group, 'NMakeBuildCommandLine').text = reconfigure_all_cmd
            ET.SubElement(all_configs_prop_group, 'NMakeReBuildCommandLine').text = reconfigure_all_cmd
            ET.SubElement(all_configs_prop_group, 'NMakeCleanCommandLine').text = ''
            ET.SubElement(all_configs_prop_group, 'ExecutablePath').text = exe_search_paths
        else:
            action = ET.SubElement(root, 'ItemDefinitionGroup')
            midl = ET.SubElement(action, 'Midl')
            ET.SubElement(midl, 'AdditionalIncludeDirectories').text = '%(AdditionalIncludeDirectories)'
            ET.SubElement(midl, 'OutputDirectory').text = '$(IntDir)'
            ET.SubElement(midl, 'HeaderFileName').text = '%(Filename).h'
            ET.SubElement(midl, 'TypeLibraryName').text = '%(Filename).tlb'
            ET.SubElement(midl, 'InterfaceIdentifierFilename').text = '%(Filename)_i.c'
            ET.SubElement(midl, 'ProxyFileName').text = '%(Filename)_p.c'
            regen_command = self.environment.get_build_command() + ['--internal', 'regencheck']
            cmd_templ = 'call %s > NUL\n"%s" "%s"'
            regen_command = cmd_templ % (self.get_vcvars_command(), '" "'.join(regen_command), self.environment.get_scratch_dir())
            self.add_custom_build(root, 'regen', regen_command, deps=self.get_regen_filelist(), outputs=[Vs2010Backend.get_regen_stampfile(self.environment.get_build_dir())], msg='Checking whether solution needs to be regenerated.')
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
        ET.SubElement(root, 'ImportGroup', Label='ExtensionTargets')
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)

    def gen_testproj(self):
        project_name = 'RUN_TESTS'
        ofname = os.path.join(self.environment.get_build_dir(), f'{project_name}.vcxproj')
        guid = self.environment.coredata.test_guid
        if self.gen_lite:
            root, type_config = self.create_basic_project(project_name, temp_dir='install-temp', guid=guid, conftype='Makefile')
            nmake_base_meson_command, exe_search_paths = Vs2010Backend.get_nmake_base_meson_command_and_exe_search_paths()
            multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list()
            _, build_dir_tail = os.path.split(self.src_to_build)
            proj_to_multiconfigured_builds_parent_dir = '..'
            for buildtype in multi_config_buildtype_list:
                meson_build_dir_for_buildtype = build_dir_tail[:-2] + buildtype
                proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
                test_cmd = f'{nmake_base_meson_command} test -C "{proj_to_build_dir_for_buildtype}" --no-rebuild'
                if not self.environment.coredata.get_option(OptionKey('stdsplit')):
                    test_cmd += ' --no-stdsplit'
                if self.environment.coredata.get_option(OptionKey('errorlogs')):
                    test_cmd += ' --print-errorlogs'
                condition = f"'$(Configuration)|$(Platform)'=='{buildtype}|{self.platform}'"
                prop_group = ET.SubElement(root, 'PropertyGroup', Condition=condition)
                ET.SubElement(prop_group, 'NMakeBuildCommandLine').text = test_cmd
                ET.SubElement(prop_group, 'ExecutablePath').text = exe_search_paths
        else:
            root, type_config = self.create_basic_project(project_name, temp_dir='test-temp', guid=guid)
            action = ET.SubElement(root, 'ItemDefinitionGroup')
            midl = ET.SubElement(action, 'Midl')
            ET.SubElement(midl, 'AdditionalIncludeDirectories').text = '%(AdditionalIncludeDirectories)'
            ET.SubElement(midl, 'OutputDirectory').text = '$(IntDir)'
            ET.SubElement(midl, 'HeaderFileName').text = '%(Filename).h'
            ET.SubElement(midl, 'TypeLibraryName').text = '%(Filename).tlb'
            ET.SubElement(midl, 'InterfaceIdentifierFilename').text = '%(Filename)_i.c'
            ET.SubElement(midl, 'ProxyFileName').text = '%(Filename)_p.c'
            test_command = self.environment.get_build_command() + ['test', '--no-rebuild']
            if not self.environment.coredata.get_option(OptionKey('stdsplit')):
                test_command += ['--no-stdsplit']
            if self.environment.coredata.get_option(OptionKey('errorlogs')):
                test_command += ['--print-errorlogs']
            self.serialize_tests()
            self.add_custom_build(root, 'run_tests', '"%s"' % '" "'.join(test_command))
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
        self.add_regen_dependency(root)
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)

    def gen_installproj(self):
        project_name = 'RUN_INSTALL'
        ofname = os.path.join(self.environment.get_build_dir(), f'{project_name}.vcxproj')
        guid = self.environment.coredata.install_guid
        if self.gen_lite:
            root, type_config = self.create_basic_project(project_name, temp_dir='install-temp', guid=guid, conftype='Makefile')
            nmake_base_meson_command, exe_search_paths = Vs2010Backend.get_nmake_base_meson_command_and_exe_search_paths()
            multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list()
            _, build_dir_tail = os.path.split(self.src_to_build)
            proj_to_multiconfigured_builds_parent_dir = '..'
            for buildtype in multi_config_buildtype_list:
                meson_build_dir_for_buildtype = build_dir_tail[:-2] + buildtype
                proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
                install_cmd = f'{nmake_base_meson_command} install -C "{proj_to_build_dir_for_buildtype}" --no-rebuild'
                condition = f"'$(Configuration)|$(Platform)'=='{buildtype}|{self.platform}'"
                prop_group = ET.SubElement(root, 'PropertyGroup', Condition=condition)
                ET.SubElement(prop_group, 'NMakeBuildCommandLine').text = install_cmd
                ET.SubElement(prop_group, 'ExecutablePath').text = exe_search_paths
        else:
            self.create_install_data_files()
            root, type_config = self.create_basic_project(project_name, temp_dir='install-temp', guid=guid)
            action = ET.SubElement(root, 'ItemDefinitionGroup')
            midl = ET.SubElement(action, 'Midl')
            ET.SubElement(midl, 'AdditionalIncludeDirectories').text = '%(AdditionalIncludeDirectories)'
            ET.SubElement(midl, 'OutputDirectory').text = '$(IntDir)'
            ET.SubElement(midl, 'HeaderFileName').text = '%(Filename).h'
            ET.SubElement(midl, 'TypeLibraryName').text = '%(Filename).tlb'
            ET.SubElement(midl, 'InterfaceIdentifierFilename').text = '%(Filename)_i.c'
            ET.SubElement(midl, 'ProxyFileName').text = '%(Filename)_p.c'
            install_command = self.environment.get_build_command() + ['install', '--no-rebuild']
            self.add_custom_build(root, 'run_install', '"%s"' % '" "'.join(install_command))
        ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
        self.add_regen_dependency(root)
        self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)

    def add_custom_build(self, node: ET.Element, rulename: str, command: str, deps: T.Optional[T.List[str]]=None, outputs: T.Optional[T.List[str]]=None, msg: T.Optional[str]=None, verify_files: bool=True) -> None:
        igroup = ET.SubElement(node, 'ItemGroup')
        rulefile = os.path.join(self.environment.get_scratch_dir(), rulename + '.rule')
        if not os.path.exists(rulefile):
            with open(rulefile, 'w', encoding='utf-8') as f:
                f.write('# Meson regen file.')
        custombuild = ET.SubElement(igroup, 'CustomBuild', Include=rulefile)
        if msg:
            message = ET.SubElement(custombuild, 'Message')
            message.text = msg
        if not verify_files:
            ET.SubElement(custombuild, 'VerifyInputsAndOutputsExist').text = 'false'
        ET.SubElement(custombuild, 'Command').text = f'{command}\n'
        if not outputs:
            outputs = [self.nonexistent_file(os.path.join(self.environment.get_scratch_dir(), 'outofdate.file'))]
        ET.SubElement(custombuild, 'Outputs').text = ';'.join(outputs)
        if deps:
            ET.SubElement(custombuild, 'AdditionalInputs').text = ';'.join(deps)

    @staticmethod
    def nonexistent_file(prefix: str) -> str:
        i = 0
        file = prefix
        while os.path.exists(file):
            file = '%s%d' % (prefix, i)
        return file

    def generate_debug_information(self, link: ET.Element) -> None:
        ET.SubElement(link, 'GenerateDebugInformation').text = 'true'

    def add_regen_dependency(self, root: ET.Element) -> None:
        if not self.gen_lite:
            regen_vcxproj = os.path.join(self.environment.get_build_dir(), 'REGEN.vcxproj')
            self.add_project_reference(root, regen_vcxproj, self.environment.coredata.regen_guid)

    def generate_lang_standard_info(self, file_args: T.Dict[str, CompilerArgs], clconf: ET.Element) -> None:
        pass