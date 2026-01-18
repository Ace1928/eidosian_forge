import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
class NinjaWriter:

    def __init__(self, hash_for_rules, target_outputs, base_dir, build_dir, output_file, toplevel_build, output_file_name, flavor, toplevel_dir=None):
        """
        base_dir: path from source root to directory containing this gyp file,
                  by gyp semantics, all input paths are relative to this
        build_dir: path from source root to build output
        toplevel_dir: path to the toplevel directory
        """
        self.hash_for_rules = hash_for_rules
        self.target_outputs = target_outputs
        self.base_dir = base_dir
        self.build_dir = build_dir
        self.ninja = ninja_syntax.Writer(output_file)
        self.toplevel_build = toplevel_build
        self.output_file_name = output_file_name
        self.flavor = flavor
        self.abs_build_dir = None
        if toplevel_dir is not None:
            self.abs_build_dir = os.path.abspath(os.path.join(toplevel_dir, build_dir))
        self.obj_ext = '.obj' if flavor == 'win' else '.o'
        if flavor == 'win':
            self.win_env = {}
            for arch in ('x86', 'x64'):
                self.win_env[arch] = 'environment.' + arch
        build_to_top = gyp.common.InvertRelativePath(build_dir, toplevel_dir)
        self.build_to_base = os.path.join(build_to_top, base_dir)
        base_to_top = gyp.common.InvertRelativePath(base_dir, toplevel_dir)
        self.base_to_build = os.path.join(base_to_top, build_dir)

    def ExpandSpecial(self, path, product_dir=None):
        """Expand specials like $!PRODUCT_DIR in |path|.

        If |product_dir| is None, assumes the cwd is already the product
        dir.  Otherwise, |product_dir| is the relative path to the product
        dir.
        """
        PRODUCT_DIR = '$!PRODUCT_DIR'
        if PRODUCT_DIR in path:
            if product_dir:
                path = path.replace(PRODUCT_DIR, product_dir)
            else:
                path = path.replace(PRODUCT_DIR + '/', '')
                path = path.replace(PRODUCT_DIR + '\\', '')
                path = path.replace(PRODUCT_DIR, '.')
        INTERMEDIATE_DIR = '$!INTERMEDIATE_DIR'
        if INTERMEDIATE_DIR in path:
            int_dir = self.GypPathToUniqueOutput('gen')
            path = path.replace(INTERMEDIATE_DIR, os.path.join(product_dir or '', int_dir))
        CONFIGURATION_NAME = '$|CONFIGURATION_NAME'
        path = path.replace(CONFIGURATION_NAME, self.config_name)
        return path

    def ExpandRuleVariables(self, path, root, dirname, source, ext, name):
        if self.flavor == 'win':
            path = self.msvs_settings.ConvertVSMacros(path, config=self.config_name)
        path = path.replace(generator_default_variables['RULE_INPUT_ROOT'], root)
        path = path.replace(generator_default_variables['RULE_INPUT_DIRNAME'], dirname)
        path = path.replace(generator_default_variables['RULE_INPUT_PATH'], source)
        path = path.replace(generator_default_variables['RULE_INPUT_EXT'], ext)
        path = path.replace(generator_default_variables['RULE_INPUT_NAME'], name)
        return path

    def GypPathToNinja(self, path, env=None):
        """Translate a gyp path to a ninja path, optionally expanding environment
        variable references in |path| with |env|.

        See the above discourse on path conversions."""
        if env:
            if self.flavor == 'mac':
                path = gyp.xcode_emulation.ExpandEnvVars(path, env)
            elif self.flavor == 'win':
                path = gyp.msvs_emulation.ExpandMacros(path, env)
        if path.startswith('$!'):
            expanded = self.ExpandSpecial(path)
            if self.flavor == 'win':
                expanded = os.path.normpath(expanded)
            return expanded
        if '$|' in path:
            path = self.ExpandSpecial(path)
        assert '$' not in path, path
        return os.path.normpath(os.path.join(self.build_to_base, path))

    def GypPathToUniqueOutput(self, path, qualified=True):
        """Translate a gyp path to a ninja path for writing output.

        If qualified is True, qualify the resulting filename with the name
        of the target.  This is necessary when e.g. compiling the same
        path twice for two separate output targets.

        See the above discourse on path conversions."""
        path = self.ExpandSpecial(path)
        assert not path.startswith('$'), path
        obj = 'obj'
        if self.toolset != 'target':
            obj += '.' + self.toolset
        path_dir, path_basename = os.path.split(path)
        assert not os.path.isabs(path_dir), "'%s' can not be absolute path (see crbug.com/462153)." % path_dir
        if qualified:
            path_basename = self.name + '.' + path_basename
        return os.path.normpath(os.path.join(obj, self.base_dir, path_dir, path_basename))

    def WriteCollapsedDependencies(self, name, targets, order_only=None):
        """Given a list of targets, return a path for a single file
        representing the result of building all the targets or None.

        Uses a stamp file if necessary."""
        assert targets == [item for item in targets if item], targets
        if len(targets) == 0:
            assert not order_only
            return None
        if len(targets) > 1 or order_only:
            stamp = self.GypPathToUniqueOutput(name + '.stamp')
            targets = self.ninja.build(stamp, 'stamp', targets, order_only=order_only)
            self.ninja.newline()
        return targets[0]

    def _SubninjaNameForArch(self, arch):
        output_file_base = os.path.splitext(self.output_file_name)[0]
        return f'{output_file_base}.{arch}.ninja'

    def WriteSpec(self, spec, config_name, generator_flags):
        """The main entry point for NinjaWriter: write the build rules for a spec.

        Returns a Target object, which represents the output paths for this spec.
        Returns None if there are no outputs (e.g. a settings-only 'none' type
        target)."""
        self.config_name = config_name
        self.name = spec['target_name']
        self.toolset = spec['toolset']
        config = spec['configurations'][config_name]
        self.target = Target(spec['type'])
        self.is_standalone_static_library = bool(spec.get('standalone_static_library', 0))
        self.target_rpath = generator_flags.get('target_rpath', '\\$$ORIGIN/lib/')
        self.is_mac_bundle = gyp.xcode_emulation.IsMacBundle(self.flavor, spec)
        self.xcode_settings = self.msvs_settings = None
        if self.flavor == 'mac':
            self.xcode_settings = gyp.xcode_emulation.XcodeSettings(spec)
            mac_toolchain_dir = generator_flags.get('mac_toolchain_dir', None)
            if mac_toolchain_dir:
                self.xcode_settings.mac_toolchain_dir = mac_toolchain_dir
        if self.flavor == 'win':
            self.msvs_settings = gyp.msvs_emulation.MsvsSettings(spec, generator_flags)
            arch = self.msvs_settings.GetArch(config_name)
            self.ninja.variable('arch', self.win_env[arch])
            self.ninja.variable('cc', '$cl_' + arch)
            self.ninja.variable('cxx', '$cl_' + arch)
            self.ninja.variable('cc_host', '$cl_' + arch)
            self.ninja.variable('cxx_host', '$cl_' + arch)
            self.ninja.variable('asm', '$ml_' + arch)
        if self.flavor == 'mac':
            self.archs = self.xcode_settings.GetActiveArchs(config_name)
            if len(self.archs) > 1:
                self.arch_subninjas = {arch: ninja_syntax.Writer(OpenOutput(os.path.join(self.toplevel_build, self._SubninjaNameForArch(arch)), 'w')) for arch in self.archs}
        actions_depends = []
        compile_depends = []
        if 'dependencies' in spec:
            for dep in spec['dependencies']:
                if dep in self.target_outputs:
                    target = self.target_outputs[dep]
                    actions_depends.append(target.PreActionInput(self.flavor))
                    compile_depends.append(target.PreCompileInput())
                    if target.uses_cpp:
                        self.target.uses_cpp = True
            actions_depends = [item for item in actions_depends if item]
            compile_depends = [item for item in compile_depends if item]
            actions_depends = self.WriteCollapsedDependencies('actions_depends', actions_depends)
            compile_depends = self.WriteCollapsedDependencies('compile_depends', compile_depends)
            self.target.preaction_stamp = actions_depends
            self.target.precompile_stamp = compile_depends
        extra_sources = []
        mac_bundle_depends = []
        self.target.actions_stamp = self.WriteActionsRulesCopies(spec, extra_sources, actions_depends, mac_bundle_depends)
        compile_depends_stamp = self.target.actions_stamp or compile_depends
        link_deps = []
        try:
            sources = extra_sources + spec.get('sources', [])
        except TypeError:
            print('extra_sources: ', str(extra_sources))
            print('spec.get("sources"): ', str(spec.get('sources')))
            raise
        if sources:
            if self.flavor == 'mac' and len(self.archs) > 1:
                for arch in self.archs:
                    self.ninja.subninja(self._SubninjaNameForArch(arch))
            pch = None
            if self.flavor == 'win':
                gyp.msvs_emulation.VerifyMissingSources(sources, self.abs_build_dir, generator_flags, self.GypPathToNinja)
                pch = gyp.msvs_emulation.PrecompiledHeader(self.msvs_settings, config_name, self.GypPathToNinja, self.GypPathToUniqueOutput, self.obj_ext)
            else:
                pch = gyp.xcode_emulation.MacPrefixHeader(self.xcode_settings, self.GypPathToNinja, lambda path, lang: self.GypPathToUniqueOutput(path + '-' + lang))
            link_deps = self.WriteSources(self.ninja, config_name, config, sources, compile_depends_stamp, pch, spec)
            obj_outputs = [f for f in sources if f.endswith(self.obj_ext)]
            if obj_outputs:
                if self.flavor != 'mac' or len(self.archs) == 1:
                    link_deps += [self.GypPathToNinja(o) for o in obj_outputs]
                else:
                    print("Warning: Actions/rules writing object files don't work with multiarch targets, dropping. (target %s)" % spec['target_name'])
        elif self.flavor == 'mac' and len(self.archs) > 1:
            link_deps = collections.defaultdict(list)
        compile_deps = self.target.actions_stamp or actions_depends
        if self.flavor == 'win' and self.target.type == 'static_library':
            self.target.component_objs = link_deps
            self.target.compile_deps = compile_deps
        output = None
        is_empty_bundle = not link_deps and (not mac_bundle_depends)
        if link_deps or self.target.actions_stamp or actions_depends:
            output = self.WriteTarget(spec, config_name, config, link_deps, compile_deps)
            if self.is_mac_bundle:
                mac_bundle_depends.append(output)
        if self.is_mac_bundle:
            output = self.WriteMacBundle(spec, mac_bundle_depends, is_empty_bundle)
        if not output:
            return None
        assert self.target.FinalOutput(), output
        return self.target

    def _WinIdlRule(self, source, prebuild, outputs):
        """Handle the implicit VS .idl rule for one source file. Fills |outputs|
        with files that are generated."""
        outdir, output, vars, flags = self.msvs_settings.GetIdlBuildData(source, self.config_name)
        outdir = self.GypPathToNinja(outdir)

        def fix_path(path, rel=None):
            path = os.path.join(outdir, path)
            dirname, basename = os.path.split(source)
            root, ext = os.path.splitext(basename)
            path = self.ExpandRuleVariables(path, root, dirname, source, ext, basename)
            if rel:
                path = os.path.relpath(path, rel)
            return path
        vars = [(name, fix_path(value, outdir)) for name, value in vars]
        output = [fix_path(p) for p in output]
        vars.append(('outdir', outdir))
        vars.append(('idlflags', flags))
        input = self.GypPathToNinja(source)
        self.ninja.build(output, 'idl', input, variables=vars, order_only=prebuild)
        outputs.extend(output)

    def WriteWinIdlFiles(self, spec, prebuild):
        """Writes rules to match MSVS's implicit idl handling."""
        assert self.flavor == 'win'
        if self.msvs_settings.HasExplicitIdlRulesOrActions(spec):
            return []
        outputs = []
        for source in filter(lambda x: x.endswith('.idl'), spec['sources']):
            self._WinIdlRule(source, prebuild, outputs)
        return outputs

    def WriteActionsRulesCopies(self, spec, extra_sources, prebuild, mac_bundle_depends):
        """Write out the Actions, Rules, and Copies steps.  Return a path
        representing the outputs of these steps."""
        outputs = []
        if self.is_mac_bundle:
            mac_bundle_resources = spec.get('mac_bundle_resources', [])[:]
        else:
            mac_bundle_resources = []
        extra_mac_bundle_resources = []
        if 'actions' in spec:
            outputs += self.WriteActions(spec['actions'], extra_sources, prebuild, extra_mac_bundle_resources)
        if 'rules' in spec:
            outputs += self.WriteRules(spec['rules'], extra_sources, prebuild, mac_bundle_resources, extra_mac_bundle_resources)
        if 'copies' in spec:
            outputs += self.WriteCopies(spec['copies'], prebuild, mac_bundle_depends)
        if 'sources' in spec and self.flavor == 'win':
            outputs += self.WriteWinIdlFiles(spec, prebuild)
        if self.xcode_settings and self.xcode_settings.IsIosFramework():
            self.WriteiOSFrameworkHeaders(spec, outputs, prebuild)
        stamp = self.WriteCollapsedDependencies('actions_rules_copies', outputs)
        if self.is_mac_bundle:
            xcassets = self.WriteMacBundleResources(extra_mac_bundle_resources + mac_bundle_resources, mac_bundle_depends)
            partial_info_plist = self.WriteMacXCassets(xcassets, mac_bundle_depends)
            self.WriteMacInfoPlist(partial_info_plist, mac_bundle_depends)
        return stamp

    def GenerateDescription(self, verb, message, fallback):
        """Generate and return a description of a build step.

        |verb| is the short summary, e.g. ACTION or RULE.
        |message| is a hand-written description, or None if not available.
        |fallback| is the gyp-level name of the step, usable as a fallback.
        """
        if self.toolset != 'target':
            verb += '(%s)' % self.toolset
        if message:
            return f'{verb} {self.ExpandSpecial(message)}'
        else:
            return f'{verb} {self.name}: {fallback}'

    def WriteActions(self, actions, extra_sources, prebuild, extra_mac_bundle_resources):
        env = self.GetToolchainEnv()
        all_outputs = []
        for action in actions:
            name = '{}_{}'.format(action['action_name'], self.hash_for_rules)
            description = self.GenerateDescription('ACTION', action.get('message', None), name)
            win_shell_flags = self.msvs_settings.GetRuleShellFlags(action) if self.flavor == 'win' else None
            args = action['action']
            depfile = action.get('depfile', None)
            if depfile:
                depfile = self.ExpandSpecial(depfile, self.base_to_build)
            pool = 'console' if int(action.get('ninja_use_console', 0)) else None
            rule_name, _ = self.WriteNewNinjaRule(name, args, description, win_shell_flags, env, pool, depfile=depfile)
            inputs = [self.GypPathToNinja(i, env) for i in action['inputs']]
            if int(action.get('process_outputs_as_sources', False)):
                extra_sources += action['outputs']
            if int(action.get('process_outputs_as_mac_bundle_resources', False)):
                extra_mac_bundle_resources += action['outputs']
            outputs = [self.GypPathToNinja(o, env) for o in action['outputs']]
            self.ninja.build(outputs, rule_name, inputs, order_only=prebuild)
            all_outputs += outputs
            self.ninja.newline()
        return all_outputs

    def WriteRules(self, rules, extra_sources, prebuild, mac_bundle_resources, extra_mac_bundle_resources):
        env = self.GetToolchainEnv()
        all_outputs = []
        for rule in rules:
            if 'action' not in rule and (not rule.get('rule_sources', [])):
                continue
            name = '{}_{}'.format(rule['rule_name'], self.hash_for_rules)
            args = rule['action']
            description = self.GenerateDescription('RULE', rule.get('message', None), ('%s ' + generator_default_variables['RULE_INPUT_PATH']) % name)
            win_shell_flags = self.msvs_settings.GetRuleShellFlags(rule) if self.flavor == 'win' else None
            pool = 'console' if int(rule.get('ninja_use_console', 0)) else None
            rule_name, args = self.WriteNewNinjaRule(name, args, description, win_shell_flags, env, pool)
            special_locals = ('source', 'root', 'dirname', 'ext', 'name')
            needed_variables = {'source'}
            for argument in args:
                for var in special_locals:
                    if '${%s}' % var in argument:
                        needed_variables.add(var)
            needed_variables = sorted(needed_variables)

            def cygwin_munge(path):
                if win_shell_flags and win_shell_flags.cygwin:
                    return path.replace('\\', '/')
                return path
            inputs = [self.GypPathToNinja(i, env) for i in rule.get('inputs', [])]
            sources = rule.get('rule_sources', [])
            num_inputs = len(inputs)
            if prebuild:
                num_inputs += 1
            if num_inputs > 2 and len(sources) > 2:
                inputs = [self.WriteCollapsedDependencies(rule['rule_name'], inputs, order_only=prebuild)]
                prebuild = []
            for source in sources:
                source = os.path.normpath(source)
                dirname, basename = os.path.split(source)
                root, ext = os.path.splitext(basename)
                outputs = [self.ExpandRuleVariables(o, root, dirname, source, ext, basename) for o in rule['outputs']]
                if int(rule.get('process_outputs_as_sources', False)):
                    extra_sources += outputs
                was_mac_bundle_resource = source in mac_bundle_resources
                if was_mac_bundle_resource or int(rule.get('process_outputs_as_mac_bundle_resources', False)):
                    extra_mac_bundle_resources += outputs
                    if was_mac_bundle_resource:
                        mac_bundle_resources.remove(source)
                extra_bindings = []
                for var in needed_variables:
                    if var == 'root':
                        extra_bindings.append(('root', cygwin_munge(root)))
                    elif var == 'dirname':
                        dirname_expanded = self.ExpandSpecial(dirname, self.base_to_build)
                        extra_bindings.append(('dirname', cygwin_munge(dirname_expanded)))
                    elif var == 'source':
                        source_expanded = self.ExpandSpecial(source, self.base_to_build)
                        extra_bindings.append(('source', cygwin_munge(source_expanded)))
                    elif var == 'ext':
                        extra_bindings.append(('ext', ext))
                    elif var == 'name':
                        extra_bindings.append(('name', cygwin_munge(basename)))
                    else:
                        assert var is None, repr(var)
                outputs = [self.GypPathToNinja(o, env) for o in outputs]
                if self.flavor == 'win':
                    extra_bindings.append(('unique_name', hashlib.md5(outputs[0]).hexdigest()))
                self.ninja.build(outputs, rule_name, self.GypPathToNinja(source), implicit=inputs, order_only=prebuild, variables=extra_bindings)
                all_outputs.extend(outputs)
        return all_outputs

    def WriteCopies(self, copies, prebuild, mac_bundle_depends):
        outputs = []
        if self.xcode_settings:
            extra_env = self.xcode_settings.GetPerTargetSettings()
            env = self.GetToolchainEnv(additional_settings=extra_env)
        else:
            env = self.GetToolchainEnv()
        for to_copy in copies:
            for path in to_copy['files']:
                path = os.path.normpath(path)
                basename = os.path.split(path)[1]
                src = self.GypPathToNinja(path, env)
                dst = self.GypPathToNinja(os.path.join(to_copy['destination'], basename), env)
                outputs += self.ninja.build(dst, 'copy', src, order_only=prebuild)
                if self.is_mac_bundle:
                    if dst.startswith(self.xcode_settings.GetBundleContentsFolderPath()):
                        mac_bundle_depends.append(dst)
        return outputs

    def WriteiOSFrameworkHeaders(self, spec, outputs, prebuild):
        """Prebuild steps to generate hmap files and copy headers to destination."""
        framework = self.ComputeMacBundleOutput()
        all_sources = spec['sources']
        copy_headers = spec['mac_framework_headers']
        output = self.GypPathToUniqueOutput('headers.hmap')
        self.xcode_settings.header_map_path = output
        all_headers = map(self.GypPathToNinja, filter(lambda x: x.endswith('.h'), all_sources))
        variables = [('framework', framework), ('copy_headers', map(self.GypPathToNinja, copy_headers))]
        outputs.extend(self.ninja.build(output, 'compile_ios_framework_headers', all_headers, variables=variables, order_only=prebuild))

    def WriteMacBundleResources(self, resources, bundle_depends):
        """Writes ninja edges for 'mac_bundle_resources'."""
        xcassets = []
        extra_env = self.xcode_settings.GetPerTargetSettings()
        env = self.GetSortedXcodeEnv(additional_settings=extra_env)
        env = self.ComputeExportEnvString(env)
        isBinary = self.xcode_settings.IsBinaryOutputFormat(self.config_name)
        for output, res in gyp.xcode_emulation.GetMacBundleResources(generator_default_variables['PRODUCT_DIR'], self.xcode_settings, map(self.GypPathToNinja, resources)):
            output = self.ExpandSpecial(output)
            if os.path.splitext(output)[-1] != '.xcassets':
                self.ninja.build(output, 'mac_tool', res, variables=[('mactool_cmd', 'copy-bundle-resource'), ('env', env), ('binary', isBinary)])
                bundle_depends.append(output)
            else:
                xcassets.append(res)
        return xcassets

    def WriteMacXCassets(self, xcassets, bundle_depends):
        """Writes ninja edges for 'mac_bundle_resources' .xcassets files.

        This add an invocation of 'actool' via the 'mac_tool.py' helper script.
        It assumes that the assets catalogs define at least one imageset and
        thus an Assets.car file will be generated in the application resources
        directory. If this is not the case, then the build will probably be done
        at each invocation of ninja."""
        if not xcassets:
            return
        extra_arguments = {}
        settings_to_arg = {'XCASSETS_APP_ICON': 'app-icon', 'XCASSETS_LAUNCH_IMAGE': 'launch-image'}
        settings = self.xcode_settings.xcode_settings[self.config_name]
        for settings_key, arg_name in settings_to_arg.items():
            value = settings.get(settings_key)
            if value:
                extra_arguments[arg_name] = value
        partial_info_plist = None
        if extra_arguments:
            partial_info_plist = self.GypPathToUniqueOutput('assetcatalog_generated_info.plist')
            extra_arguments['output-partial-info-plist'] = partial_info_plist
        outputs = []
        outputs.append(os.path.join(self.xcode_settings.GetBundleResourceFolder(), 'Assets.car'))
        if partial_info_plist:
            outputs.append(partial_info_plist)
        keys = QuoteShellArgument(json.dumps(extra_arguments), self.flavor)
        extra_env = self.xcode_settings.GetPerTargetSettings()
        env = self.GetSortedXcodeEnv(additional_settings=extra_env)
        env = self.ComputeExportEnvString(env)
        bundle_depends.extend(self.ninja.build(outputs, 'compile_xcassets', xcassets, variables=[('env', env), ('keys', keys)]))
        return partial_info_plist

    def WriteMacInfoPlist(self, partial_info_plist, bundle_depends):
        """Write build rules for bundle Info.plist files."""
        info_plist, out, defines, extra_env = gyp.xcode_emulation.GetMacInfoPlist(generator_default_variables['PRODUCT_DIR'], self.xcode_settings, self.GypPathToNinja)
        if not info_plist:
            return
        out = self.ExpandSpecial(out)
        if defines:
            intermediate_plist = self.GypPathToUniqueOutput(os.path.basename(info_plist))
            defines = ' '.join([Define(d, self.flavor) for d in defines])
            info_plist = self.ninja.build(intermediate_plist, 'preprocess_infoplist', info_plist, variables=[('defines', defines)])
        env = self.GetSortedXcodeEnv(additional_settings=extra_env)
        env = self.ComputeExportEnvString(env)
        if partial_info_plist:
            intermediate_plist = self.GypPathToUniqueOutput('merged_info.plist')
            info_plist = self.ninja.build(intermediate_plist, 'merge_infoplist', [partial_info_plist, info_plist])
        keys = self.xcode_settings.GetExtraPlistItems(self.config_name)
        keys = QuoteShellArgument(json.dumps(keys), self.flavor)
        isBinary = self.xcode_settings.IsBinaryOutputFormat(self.config_name)
        self.ninja.build(out, 'copy_infoplist', info_plist, variables=[('env', env), ('keys', keys), ('binary', isBinary)])
        bundle_depends.append(out)

    def WriteSources(self, ninja_file, config_name, config, sources, predepends, precompiled_header, spec):
        """Write build rules to compile all of |sources|."""
        if self.toolset == 'host':
            self.ninja.variable('ar', '$ar_host')
            self.ninja.variable('cc', '$cc_host')
            self.ninja.variable('cxx', '$cxx_host')
            self.ninja.variable('ld', '$ld_host')
            self.ninja.variable('ldxx', '$ldxx_host')
            self.ninja.variable('nm', '$nm_host')
            self.ninja.variable('readelf', '$readelf_host')
        if self.flavor != 'mac' or len(self.archs) == 1:
            return self.WriteSourcesForArch(self.ninja, config_name, config, sources, predepends, precompiled_header, spec)
        else:
            return {arch: self.WriteSourcesForArch(self.arch_subninjas[arch], config_name, config, sources, predepends, precompiled_header, spec, arch=arch) for arch in self.archs}

    def WriteSourcesForArch(self, ninja_file, config_name, config, sources, predepends, precompiled_header, spec, arch=None):
        """Write build rules to compile all of |sources|."""
        extra_defines = []
        if self.flavor == 'mac':
            cflags = self.xcode_settings.GetCflags(config_name, arch=arch)
            cflags_c = self.xcode_settings.GetCflagsC(config_name)
            cflags_cc = self.xcode_settings.GetCflagsCC(config_name)
            cflags_objc = ['$cflags_c'] + self.xcode_settings.GetCflagsObjC(config_name)
            cflags_objcc = ['$cflags_cc'] + self.xcode_settings.GetCflagsObjCC(config_name)
        elif self.flavor == 'win':
            asmflags = self.msvs_settings.GetAsmflags(config_name)
            cflags = self.msvs_settings.GetCflags(config_name)
            cflags_c = self.msvs_settings.GetCflagsC(config_name)
            cflags_cc = self.msvs_settings.GetCflagsCC(config_name)
            extra_defines = self.msvs_settings.GetComputedDefines(config_name)
            pdbpath_c = pdbpath_cc = self.msvs_settings.GetCompilerPdbName(config_name, self.ExpandSpecial)
            if not pdbpath_c:
                obj = 'obj'
                if self.toolset != 'target':
                    obj += '.' + self.toolset
                pdbpath = os.path.normpath(os.path.join(obj, self.base_dir, self.name))
                pdbpath_c = pdbpath + '.c.pdb'
                pdbpath_cc = pdbpath + '.cc.pdb'
            self.WriteVariableList(ninja_file, 'pdbname_c', [pdbpath_c])
            self.WriteVariableList(ninja_file, 'pdbname_cc', [pdbpath_cc])
            self.WriteVariableList(ninja_file, 'pchprefix', [self.name])
        else:
            cflags = config.get('cflags', [])
            cflags_c = config.get('cflags_c', [])
            cflags_cc = config.get('cflags_cc', [])
        if self.toolset == 'target':
            cflags_c = os.environ.get('CPPFLAGS', '').split() + os.environ.get('CFLAGS', '').split() + cflags_c
            cflags_cc = os.environ.get('CPPFLAGS', '').split() + os.environ.get('CXXFLAGS', '').split() + cflags_cc
        elif self.toolset == 'host':
            cflags_c = os.environ.get('CPPFLAGS_host', '').split() + os.environ.get('CFLAGS_host', '').split() + cflags_c
            cflags_cc = os.environ.get('CPPFLAGS_host', '').split() + os.environ.get('CXXFLAGS_host', '').split() + cflags_cc
        defines = config.get('defines', []) + extra_defines
        self.WriteVariableList(ninja_file, 'defines', [Define(d, self.flavor) for d in defines])
        if self.flavor == 'win':
            self.WriteVariableList(ninja_file, 'asmflags', map(self.ExpandSpecial, asmflags))
            self.WriteVariableList(ninja_file, 'rcflags', [QuoteShellArgument(self.ExpandSpecial(f), self.flavor) for f in self.msvs_settings.GetRcflags(config_name, self.GypPathToNinja)])
        include_dirs = config.get('include_dirs', [])
        env = self.GetToolchainEnv()
        if self.flavor == 'win':
            include_dirs = self.msvs_settings.AdjustIncludeDirs(include_dirs, config_name)
        self.WriteVariableList(ninja_file, 'includes', [QuoteShellArgument('-I' + self.GypPathToNinja(i, env), self.flavor) for i in include_dirs])
        if self.flavor == 'win':
            midl_include_dirs = config.get('midl_include_dirs', [])
            midl_include_dirs = self.msvs_settings.AdjustMidlIncludeDirs(midl_include_dirs, config_name)
            self.WriteVariableList(ninja_file, 'midl_includes', [QuoteShellArgument('-I' + self.GypPathToNinja(i, env), self.flavor) for i in midl_include_dirs])
        pch_commands = precompiled_header.GetPchBuildCommands(arch)
        if self.flavor == 'mac':
            for ext, var in [('c', 'cflags_pch_c'), ('cc', 'cflags_pch_cc'), ('m', 'cflags_pch_objc'), ('mm', 'cflags_pch_objcc')]:
                include = precompiled_header.GetInclude(ext, arch)
                if include:
                    ninja_file.variable(var, include)
        arflags = config.get('arflags', [])
        self.WriteVariableList(ninja_file, 'cflags', map(self.ExpandSpecial, cflags))
        self.WriteVariableList(ninja_file, 'cflags_c', map(self.ExpandSpecial, cflags_c))
        self.WriteVariableList(ninja_file, 'cflags_cc', map(self.ExpandSpecial, cflags_cc))
        if self.flavor == 'mac':
            self.WriteVariableList(ninja_file, 'cflags_objc', map(self.ExpandSpecial, cflags_objc))
            self.WriteVariableList(ninja_file, 'cflags_objcc', map(self.ExpandSpecial, cflags_objcc))
        self.WriteVariableList(ninja_file, 'arflags', map(self.ExpandSpecial, arflags))
        ninja_file.newline()
        outputs = []
        has_rc_source = False
        for source in sources:
            filename, ext = os.path.splitext(source)
            ext = ext[1:]
            obj_ext = self.obj_ext
            if ext in ('cc', 'cpp', 'cxx'):
                command = 'cxx'
                self.target.uses_cpp = True
            elif ext == 'c' or (ext == 'S' and self.flavor != 'win'):
                command = 'cc'
            elif ext == 's' and self.flavor != 'win':
                command = 'cc_s'
            elif self.flavor == 'win' and ext in ('asm', 'S') and (not self.msvs_settings.HasExplicitAsmRules(spec)):
                command = 'asm'
                obj_ext = '_asm.obj'
            elif self.flavor == 'mac' and ext == 'm':
                command = 'objc'
            elif self.flavor == 'mac' and ext == 'mm':
                command = 'objcxx'
                self.target.uses_cpp = True
            elif self.flavor == 'win' and ext == 'rc':
                command = 'rc'
                obj_ext = '.res'
                has_rc_source = True
            else:
                continue
            input = self.GypPathToNinja(source)
            output = self.GypPathToUniqueOutput(filename + obj_ext)
            if arch is not None:
                output = AddArch(output, arch)
            implicit = precompiled_header.GetObjDependencies([input], [output], arch)
            variables = []
            if self.flavor == 'win':
                variables, output, implicit = precompiled_header.GetFlagsModifications(input, output, implicit, command, cflags_c, cflags_cc, self.ExpandSpecial)
            ninja_file.build(output, command, input, implicit=[gch for _, _, gch in implicit], order_only=predepends, variables=variables)
            outputs.append(output)
        if has_rc_source:
            resource_include_dirs = config.get('resource_include_dirs', include_dirs)
            self.WriteVariableList(ninja_file, 'resource_includes', [QuoteShellArgument('-I' + self.GypPathToNinja(i, env), self.flavor) for i in resource_include_dirs])
        self.WritePchTargets(ninja_file, pch_commands)
        ninja_file.newline()
        return outputs

    def WritePchTargets(self, ninja_file, pch_commands):
        """Writes ninja rules to compile prefix headers."""
        if not pch_commands:
            return
        for gch, lang_flag, lang, input in pch_commands:
            var_name = {'c': 'cflags_pch_c', 'cc': 'cflags_pch_cc', 'm': 'cflags_pch_objc', 'mm': 'cflags_pch_objcc'}[lang]
            map = {'c': 'cc', 'cc': 'cxx', 'm': 'objc', 'mm': 'objcxx'}
            cmd = map.get(lang)
            ninja_file.build(gch, cmd, input, variables=[(var_name, lang_flag)])

    def WriteLink(self, spec, config_name, config, link_deps, compile_deps):
        """Write out a link step. Fills out target.binary. """
        if self.flavor != 'mac' or len(self.archs) == 1:
            return self.WriteLinkForArch(self.ninja, spec, config_name, config, link_deps, compile_deps)
        else:
            output = self.ComputeOutput(spec)
            inputs = [self.WriteLinkForArch(self.arch_subninjas[arch], spec, config_name, config, link_deps[arch], compile_deps, arch=arch) for arch in self.archs]
            extra_bindings = []
            build_output = output
            if not self.is_mac_bundle:
                self.AppendPostbuildVariable(extra_bindings, spec, output, output)
            if spec['type'] in ('shared_library', 'loadable_module') and (not self.is_mac_bundle):
                extra_bindings.append(('lib', output))
                self.ninja.build([output, output + '.TOC'], 'solipo', inputs, variables=extra_bindings)
            else:
                self.ninja.build(build_output, 'lipo', inputs, variables=extra_bindings)
            return output

    def WriteLinkForArch(self, ninja_file, spec, config_name, config, link_deps, compile_deps, arch=None):
        """Write out a link step. Fills out target.binary. """
        command = {'executable': 'link', 'loadable_module': 'solink_module', 'shared_library': 'solink'}[spec['type']]
        command_suffix = ''
        implicit_deps = set()
        solibs = set()
        order_deps = set()
        if compile_deps:
            order_deps.add(compile_deps)
        if 'dependencies' in spec:
            extra_link_deps = set()
            for dep in spec['dependencies']:
                target = self.target_outputs.get(dep)
                if not target:
                    continue
                linkable = target.Linkable()
                if linkable:
                    new_deps = []
                    if self.flavor == 'win' and target.component_objs and self.msvs_settings.IsUseLibraryDependencyInputs(config_name):
                        new_deps = target.component_objs
                        if target.compile_deps:
                            order_deps.add(target.compile_deps)
                    elif self.flavor == 'win' and target.import_lib:
                        new_deps = [target.import_lib]
                    elif target.UsesToc(self.flavor):
                        solibs.add(target.binary)
                        implicit_deps.add(target.binary + '.TOC')
                    else:
                        new_deps = [target.binary]
                    for new_dep in new_deps:
                        if new_dep not in extra_link_deps:
                            extra_link_deps.add(new_dep)
                            link_deps.append(new_dep)
                final_output = target.FinalOutput()
                if not linkable or final_output != target.binary:
                    implicit_deps.add(final_output)
        extra_bindings = []
        if self.target.uses_cpp and self.flavor != 'win':
            extra_bindings.append(('ld', '$ldxx'))
        output = self.ComputeOutput(spec, arch)
        if arch is None and (not self.is_mac_bundle):
            self.AppendPostbuildVariable(extra_bindings, spec, output, output)
        is_executable = spec['type'] == 'executable'
        if self.toolset == 'target':
            env_ldflags = os.environ.get('LDFLAGS', '').split()
        elif self.toolset == 'host':
            env_ldflags = os.environ.get('LDFLAGS_host', '').split()
        if self.flavor == 'mac':
            ldflags = self.xcode_settings.GetLdflags(config_name, self.ExpandSpecial(generator_default_variables['PRODUCT_DIR']), self.GypPathToNinja, arch)
            ldflags = env_ldflags + ldflags
        elif self.flavor == 'win':
            manifest_base_name = self.GypPathToUniqueOutput(self.ComputeOutputFileName(spec))
            ldflags, intermediate_manifest, manifest_files = self.msvs_settings.GetLdflags(config_name, self.GypPathToNinja, self.ExpandSpecial, manifest_base_name, output, is_executable, self.toplevel_build)
            ldflags = env_ldflags + ldflags
            self.WriteVariableList(ninja_file, 'manifests', manifest_files)
            implicit_deps = implicit_deps.union(manifest_files)
            if intermediate_manifest:
                self.WriteVariableList(ninja_file, 'intermediatemanifest', [intermediate_manifest])
            command_suffix = _GetWinLinkRuleNameSuffix(self.msvs_settings.IsEmbedManifest(config_name))
            def_file = self.msvs_settings.GetDefFile(self.GypPathToNinja)
            if def_file:
                implicit_deps.add(def_file)
        else:
            ldflags = env_ldflags + config.get('ldflags', [])
            if is_executable and len(solibs):
                rpath = 'lib/'
                if self.toolset != 'target':
                    rpath += self.toolset
                    ldflags.append('-Wl,-rpath=\\$$ORIGIN/%s' % rpath)
                else:
                    ldflags.append('-Wl,-rpath=%s' % self.target_rpath)
                ldflags.append('-Wl,-rpath-link=%s' % rpath)
        self.WriteVariableList(ninja_file, 'ldflags', map(self.ExpandSpecial, ldflags))
        library_dirs = config.get('library_dirs', [])
        if self.flavor == 'win':
            library_dirs = [self.msvs_settings.ConvertVSMacros(library_dir, config_name) for library_dir in library_dirs]
            library_dirs = ['/LIBPATH:' + QuoteShellArgument(self.GypPathToNinja(library_dir), self.flavor) for library_dir in library_dirs]
        else:
            library_dirs = [QuoteShellArgument('-L' + self.GypPathToNinja(library_dir), self.flavor) for library_dir in library_dirs]
        libraries = gyp.common.uniquer(map(self.ExpandSpecial, spec.get('libraries', [])))
        if self.flavor == 'mac':
            libraries = self.xcode_settings.AdjustLibraries(libraries, config_name)
        elif self.flavor == 'win':
            libraries = self.msvs_settings.AdjustLibraries(libraries)
        self.WriteVariableList(ninja_file, 'libs', library_dirs + libraries)
        linked_binary = output
        if command in ('solink', 'solink_module'):
            extra_bindings.append(('soname', os.path.split(output)[1]))
            extra_bindings.append(('lib', gyp.common.EncodePOSIXShellArgument(output)))
            if self.flavor != 'win':
                link_file_list = output
                if self.is_mac_bundle:
                    link_file_list = self.xcode_settings.GetWrapperName()
                if arch:
                    link_file_list += '.' + arch
                link_file_list += '.rsp'
                link_file_list = link_file_list.replace(' ', '_')
                extra_bindings.append(('link_file_list', gyp.common.EncodePOSIXShellArgument(link_file_list)))
            if self.flavor == 'win':
                extra_bindings.append(('binary', output))
                if '/NOENTRY' not in ldflags and (not self.msvs_settings.GetNoImportLibrary(config_name)):
                    self.target.import_lib = output + '.lib'
                    extra_bindings.append(('implibflag', '/IMPLIB:%s' % self.target.import_lib))
                    pdbname = self.msvs_settings.GetPDBName(config_name, self.ExpandSpecial, output + '.pdb')
                    output = [output, self.target.import_lib]
                    if pdbname:
                        output.append(pdbname)
            elif not self.is_mac_bundle:
                output = [output, output + '.TOC']
            else:
                command = command + '_notoc'
        elif self.flavor == 'win':
            extra_bindings.append(('binary', output))
            pdbname = self.msvs_settings.GetPDBName(config_name, self.ExpandSpecial, output + '.pdb')
            if pdbname:
                output = [output, pdbname]
        if len(solibs):
            extra_bindings.append(('solibs', gyp.common.EncodePOSIXShellList(sorted(solibs))))
        ninja_file.build(output, command + command_suffix, link_deps, implicit=sorted(implicit_deps), order_only=list(order_deps), variables=extra_bindings)
        return linked_binary

    def WriteTarget(self, spec, config_name, config, link_deps, compile_deps):
        extra_link_deps = any((self.target_outputs.get(dep).Linkable() for dep in spec.get('dependencies', []) if dep in self.target_outputs))
        if spec['type'] == 'none' or (not link_deps and (not extra_link_deps)):
            self.target.binary = compile_deps
            self.target.type = 'none'
        elif spec['type'] == 'static_library':
            self.target.binary = self.ComputeOutput(spec)
            if self.flavor not in ('mac', 'openbsd', 'netbsd', 'win') and (not self.is_standalone_static_library):
                self.ninja.build(self.target.binary, 'alink_thin', link_deps, order_only=compile_deps)
            else:
                variables = []
                if self.xcode_settings:
                    libtool_flags = self.xcode_settings.GetLibtoolflags(config_name)
                    if libtool_flags:
                        variables.append(('libtool_flags', libtool_flags))
                if self.msvs_settings:
                    libflags = self.msvs_settings.GetLibFlags(config_name, self.GypPathToNinja)
                    variables.append(('libflags', libflags))
                if self.flavor != 'mac' or len(self.archs) == 1:
                    self.AppendPostbuildVariable(variables, spec, self.target.binary, self.target.binary)
                    self.ninja.build(self.target.binary, 'alink', link_deps, order_only=compile_deps, variables=variables)
                else:
                    inputs = []
                    for arch in self.archs:
                        output = self.ComputeOutput(spec, arch)
                        self.arch_subninjas[arch].build(output, 'alink', link_deps[arch], order_only=compile_deps, variables=variables)
                        inputs.append(output)
                    self.AppendPostbuildVariable(variables, spec, self.target.binary, self.target.binary)
                    self.ninja.build(self.target.binary, 'alink', inputs, variables=variables)
        else:
            self.target.binary = self.WriteLink(spec, config_name, config, link_deps, compile_deps)
        return self.target.binary

    def WriteMacBundle(self, spec, mac_bundle_depends, is_empty):
        assert self.is_mac_bundle
        package_framework = spec['type'] in ('shared_library', 'loadable_module')
        output = self.ComputeMacBundleOutput()
        if is_empty:
            output += '.stamp'
        variables = []
        self.AppendPostbuildVariable(variables, spec, output, self.target.binary, is_command_start=not package_framework)
        if package_framework and (not is_empty):
            if spec['type'] == 'shared_library' and self.xcode_settings.isIOS:
                self.ninja.build(output, 'package_ios_framework', mac_bundle_depends, variables=variables)
            else:
                variables.append(('version', self.xcode_settings.GetFrameworkVersion()))
                self.ninja.build(output, 'package_framework', mac_bundle_depends, variables=variables)
        else:
            self.ninja.build(output, 'stamp', mac_bundle_depends, variables=variables)
        self.target.bundle = output
        return output

    def GetToolchainEnv(self, additional_settings=None):
        """Returns the variables toolchain would set for build steps."""
        env = self.GetSortedXcodeEnv(additional_settings=additional_settings)
        if self.flavor == 'win':
            env = self.GetMsvsToolchainEnv(additional_settings=additional_settings)
        return env

    def GetMsvsToolchainEnv(self, additional_settings=None):
        """Returns the variables Visual Studio would set for build steps."""
        return self.msvs_settings.GetVSMacroEnv('$!PRODUCT_DIR', config=self.config_name)

    def GetSortedXcodeEnv(self, additional_settings=None):
        """Returns the variables Xcode would set for build steps."""
        assert self.abs_build_dir
        abs_build_dir = self.abs_build_dir
        return gyp.xcode_emulation.GetSortedXcodeEnv(self.xcode_settings, abs_build_dir, os.path.join(abs_build_dir, self.build_to_base), self.config_name, additional_settings)

    def GetSortedXcodePostbuildEnv(self):
        """Returns the variables Xcode would set for postbuild steps."""
        postbuild_settings = {}
        strip_save_file = self.xcode_settings.GetPerTargetSetting('CHROMIUM_STRIP_SAVE_FILE')
        if strip_save_file:
            postbuild_settings['CHROMIUM_STRIP_SAVE_FILE'] = strip_save_file
        return self.GetSortedXcodeEnv(additional_settings=postbuild_settings)

    def AppendPostbuildVariable(self, variables, spec, output, binary, is_command_start=False):
        """Adds a 'postbuild' variable if there is a postbuild for |output|."""
        postbuild = self.GetPostbuildCommand(spec, output, binary, is_command_start)
        if postbuild:
            variables.append(('postbuilds', postbuild))

    def GetPostbuildCommand(self, spec, output, output_binary, is_command_start):
        """Returns a shell command that runs all the postbuilds, and removes
        |output| if any of them fails. If |is_command_start| is False, then the
        returned string will start with ' && '."""
        if not self.xcode_settings or spec['type'] == 'none' or (not output):
            return ''
        output = QuoteShellArgument(output, self.flavor)
        postbuilds = gyp.xcode_emulation.GetSpecPostbuildCommands(spec, quiet=True)
        if output_binary is not None:
            postbuilds = self.xcode_settings.AddImplicitPostbuilds(self.config_name, os.path.normpath(os.path.join(self.base_to_build, output)), QuoteShellArgument(os.path.normpath(os.path.join(self.base_to_build, output_binary)), self.flavor), postbuilds, quiet=True)
        if not postbuilds:
            return ''
        postbuilds.insert(0, gyp.common.EncodePOSIXShellList(['cd', self.build_to_base]))
        env = self.ComputeExportEnvString(self.GetSortedXcodePostbuildEnv())
        commands = env + ' (' + ' && '.join([ninja_syntax.escape(command) for command in postbuilds])
        command_string = commands + '); G=$$?; ((exit $$G) || rm -rf %s) ' % output + '&& exit $$G)'
        if is_command_start:
            return '(' + command_string + ' && '
        else:
            return '$ && (' + command_string

    def ComputeExportEnvString(self, env):
        """Given an environment, returns a string looking like
            'export FOO=foo; export BAR="${FOO} bar;'
        that exports |env| to the shell."""
        export_str = []
        for k, v in env:
            export_str.append('export %s=%s;' % (k, ninja_syntax.escape(gyp.common.EncodePOSIXShellArgument(v))))
        return ' '.join(export_str)

    def ComputeMacBundleOutput(self):
        """Return the 'output' (full output path) to a bundle output directory."""
        assert self.is_mac_bundle
        path = generator_default_variables['PRODUCT_DIR']
        return self.ExpandSpecial(os.path.join(path, self.xcode_settings.GetWrapperName()))

    def ComputeOutputFileName(self, spec, type=None):
        """Compute the filename of the final output for the current target."""
        if not type:
            type = spec['type']
        default_variables = copy.copy(generator_default_variables)
        CalculateVariables(default_variables, {'flavor': self.flavor})
        DEFAULT_PREFIX = {'loadable_module': default_variables['SHARED_LIB_PREFIX'], 'shared_library': default_variables['SHARED_LIB_PREFIX'], 'static_library': default_variables['STATIC_LIB_PREFIX'], 'executable': default_variables['EXECUTABLE_PREFIX']}
        prefix = spec.get('product_prefix', DEFAULT_PREFIX.get(type, ''))
        DEFAULT_EXTENSION = {'loadable_module': default_variables['SHARED_LIB_SUFFIX'], 'shared_library': default_variables['SHARED_LIB_SUFFIX'], 'static_library': default_variables['STATIC_LIB_SUFFIX'], 'executable': default_variables['EXECUTABLE_SUFFIX']}
        extension = spec.get('product_extension')
        if extension:
            extension = '.' + extension
        else:
            extension = DEFAULT_EXTENSION.get(type, '')
        if 'product_name' in spec:
            target = spec['product_name']
        else:
            target = spec['target_name']
            if prefix == 'lib':
                target = StripPrefix(target, 'lib')
        if type in ('static_library', 'loadable_module', 'shared_library', 'executable'):
            return f'{prefix}{target}{extension}'
        elif type == 'none':
            return '%s.stamp' % target
        else:
            raise Exception('Unhandled output type %s' % type)

    def ComputeOutput(self, spec, arch=None):
        """Compute the path for the final output of the spec."""
        type = spec['type']
        if self.flavor == 'win':
            override = self.msvs_settings.GetOutputName(self.config_name, self.ExpandSpecial)
            if override:
                return override
        if arch is None and self.flavor == 'mac' and (type in ('static_library', 'executable', 'shared_library', 'loadable_module')):
            filename = self.xcode_settings.GetExecutablePath()
        else:
            filename = self.ComputeOutputFileName(spec, type)
        if arch is None and 'product_dir' in spec:
            path = os.path.join(spec['product_dir'], filename)
            return self.ExpandSpecial(path)
        type_in_output_root = ['executable', 'loadable_module']
        if self.flavor == 'mac' and self.toolset == 'target':
            type_in_output_root += ['shared_library', 'static_library']
        elif self.flavor == 'win' and self.toolset == 'target':
            type_in_output_root += ['shared_library']
        if arch is not None:
            archdir = 'arch'
            if self.toolset != 'target':
                archdir = os.path.join('arch', '%s' % self.toolset)
            return os.path.join(archdir, AddArch(filename, arch))
        elif type in type_in_output_root or self.is_standalone_static_library:
            return filename
        elif type == 'shared_library':
            libdir = 'lib'
            if self.toolset != 'target':
                libdir = os.path.join('lib', '%s' % self.toolset)
            return os.path.join(libdir, filename)
        else:
            return self.GypPathToUniqueOutput(filename, qualified=False)

    def WriteVariableList(self, ninja_file, var, values):
        assert not isinstance(values, str)
        if values is None:
            values = []
        ninja_file.variable(var, ' '.join(values))

    def WriteNewNinjaRule(self, name, args, description, win_shell_flags, env, pool, depfile=None):
        """Write out a new ninja "rule" statement for a given command.

        Returns the name of the new rule, and a copy of |args| with variables
        expanded."""
        if self.flavor == 'win':
            args = [self.msvs_settings.ConvertVSMacros(arg, self.base_to_build, config=self.config_name) for arg in args]
            description = self.msvs_settings.ConvertVSMacros(description, config=self.config_name)
        elif self.flavor == 'mac':
            args = [gyp.xcode_emulation.ExpandEnvVars(arg, env) for arg in args]
            description = gyp.xcode_emulation.ExpandEnvVars(description, env)
        rule_name = self.name
        if self.toolset == 'target':
            rule_name += '.' + self.toolset
        rule_name += '.' + name
        rule_name = re.sub('[^a-zA-Z0-9_]', '_', rule_name)
        protect = ['${root}', '${dirname}', '${source}', '${ext}', '${name}']
        protect = '(?!' + '|'.join(map(re.escape, protect)) + ')'
        description = re.sub(protect + '\\$', '_', description)
        rspfile = None
        rspfile_content = None
        args = [self.ExpandSpecial(arg, self.base_to_build) for arg in args]
        if self.flavor == 'win':
            rspfile = rule_name + '.$unique_name.rsp'
            run_in = '' if win_shell_flags.cygwin else ' ' + self.build_to_base
            if win_shell_flags.cygwin:
                rspfile_content = self.msvs_settings.BuildCygwinBashCommandLine(args, self.build_to_base)
            else:
                rspfile_content = gyp.msvs_emulation.EncodeRspFileList(args, win_shell_flags.quote)
            command = '%s gyp-win-tool action-wrapper $arch ' % sys.executable + rspfile + run_in
        else:
            env = self.ComputeExportEnvString(env)
            command = gyp.common.EncodePOSIXShellList(args)
            command = 'cd %s; ' % self.build_to_base + env + command
        self.ninja.rule(rule_name, command, description, depfile=depfile, restat=True, pool=pool, rspfile=rspfile, rspfile_content=rspfile_content)
        self.ninja.newline()
        return (rule_name, args)