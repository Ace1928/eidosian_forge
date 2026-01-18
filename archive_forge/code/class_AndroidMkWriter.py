import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
class AndroidMkWriter:
    """AndroidMkWriter packages up the writing of one target-specific Android.mk.

    Its only real entry point is Write(), and is mostly used for namespacing.
    """

    def __init__(self, android_top_dir):
        self.android_top_dir = android_top_dir

    def Write(self, qualified_target, relative_target, base_path, output_filename, spec, configs, part_of_all, write_alias_target, sdk_version):
        """The main entry point: writes a .mk file for a single target.

        Arguments:
          qualified_target: target we're generating
          relative_target: qualified target name relative to the root
          base_path: path relative to source root we're building in, used to resolve
                     target-relative paths
          output_filename: output .mk file name to write
          spec, configs: gyp info
          part_of_all: flag indicating this target is part of 'all'
          write_alias_target: flag indicating whether to create short aliases for
                              this target
          sdk_version: what to emit for LOCAL_SDK_VERSION in output
        """
        gyp.common.EnsureDirExists(output_filename)
        self.fp = open(output_filename, 'w')
        self.fp.write(header)
        self.qualified_target = qualified_target
        self.relative_target = relative_target
        self.path = base_path
        self.target = spec['target_name']
        self.type = spec['type']
        self.toolset = spec['toolset']
        deps, link_deps = self.ComputeDeps(spec)
        extra_outputs = []
        extra_sources = []
        self.android_class = MODULE_CLASSES.get(self.type, 'GYP')
        self.android_module = self.ComputeAndroidModule(spec)
        self.android_stem, self.android_suffix = self.ComputeOutputParts(spec)
        self.output = self.output_binary = self.ComputeOutput(spec)
        self.WriteLn('include $(CLEAR_VARS)\n')
        self.WriteLn('LOCAL_MODULE_CLASS := ' + self.android_class)
        self.WriteLn('LOCAL_MODULE := ' + self.android_module)
        if self.android_stem != self.android_module:
            self.WriteLn('LOCAL_MODULE_STEM := ' + self.android_stem)
        self.WriteLn('LOCAL_MODULE_SUFFIX := ' + self.android_suffix)
        if self.toolset == 'host':
            self.WriteLn('LOCAL_IS_HOST_MODULE := true')
            self.WriteLn('LOCAL_MULTILIB := $(GYP_HOST_MULTILIB)')
        elif sdk_version > 0:
            self.WriteLn('LOCAL_MODULE_TARGET_ARCH := $(TARGET_$(GYP_VAR_PREFIX)ARCH)')
            self.WriteLn('LOCAL_SDK_VERSION := %s' % sdk_version)
        if self.toolset == 'host':
            self.WriteLn('gyp_intermediate_dir := $(call local-intermediates-dir,,$(GYP_HOST_VAR_PREFIX))')
        else:
            self.WriteLn('gyp_intermediate_dir := $(call local-intermediates-dir,,$(GYP_VAR_PREFIX))')
        self.WriteLn('gyp_shared_intermediate_dir := $(call intermediates-dir-for,GYP,shared,,,$(GYP_VAR_PREFIX))')
        self.WriteLn()
        target_dependencies = [x[1] for x in deps if x[0] == 'path']
        self.WriteLn('# Make sure our deps are built first.')
        self.WriteList(target_dependencies, 'GYP_TARGET_DEPENDENCIES', local_pathify=True)
        if 'actions' in spec:
            self.WriteActions(spec['actions'], extra_sources, extra_outputs)
        if 'rules' in spec:
            self.WriteRules(spec['rules'], extra_sources, extra_outputs)
        if 'copies' in spec:
            self.WriteCopies(spec['copies'], extra_outputs)
        self.WriteList(extra_outputs, 'GYP_GENERATED_OUTPUTS', local_pathify=True)
        self.WriteLn('# Make sure our deps and generated files are built first.')
        self.WriteLn('LOCAL_ADDITIONAL_DEPENDENCIES := $(GYP_TARGET_DEPENDENCIES) $(GYP_GENERATED_OUTPUTS)')
        self.WriteLn()
        if spec.get('sources', []) or extra_sources:
            self.WriteSources(spec, configs, extra_sources)
        self.WriteTarget(spec, configs, deps, link_deps, part_of_all, write_alias_target)
        target_outputs[qualified_target] = ('path', self.output_binary)
        if self.type == 'static_library':
            target_link_deps[qualified_target] = ('static', self.android_module)
        elif self.type == 'shared_library':
            target_link_deps[qualified_target] = ('shared', self.android_module)
        self.fp.close()
        return self.android_module

    def WriteActions(self, actions, extra_sources, extra_outputs):
        """Write Makefile code for any 'actions' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       actions (used to make other pieces dependent on these
                       actions)
        """
        for action in actions:
            name = make.StringToMakefileVariable('{}_{}'.format(self.relative_target, action['action_name']))
            self.WriteLn('### Rules for action "%s":' % action['action_name'])
            inputs = action['inputs']
            outputs = action['outputs']
            dirs = set()
            for out in outputs:
                if not out.startswith('$'):
                    print('WARNING: Action for target "%s" writes output to local path "%s".' % (self.target, out))
                dir = os.path.split(out)[0]
                if dir:
                    dirs.add(dir)
            if int(action.get('process_outputs_as_sources', False)):
                extra_sources += outputs
            command = gyp.common.EncodePOSIXShellList(action['action'])
            if 'message' in action:
                quiet_cmd = 'Gyp action: %s ($@)' % action['message']
            else:
                quiet_cmd = 'Gyp action: %s ($@)' % name
            if len(dirs) > 0:
                command = 'mkdir -p %s' % ' '.join(dirs) + '; ' + command
            cd_action = 'cd $(gyp_local_path)/%s; ' % self.path
            command = cd_action + command
            main_output = make.QuoteSpaces(self.LocalPathify(outputs[0]))
            self.WriteLn('%s: gyp_local_path := $(LOCAL_PATH)' % main_output)
            self.WriteLn('%s: gyp_var_prefix := $(GYP_VAR_PREFIX)' % main_output)
            self.WriteLn('%s: gyp_intermediate_dir := $(abspath $(gyp_intermediate_dir))' % main_output)
            self.WriteLn('%s: gyp_shared_intermediate_dir := $(abspath $(gyp_shared_intermediate_dir))' % main_output)
            self.WriteLn('%s: export PATH := $(subst $(ANDROID_BUILD_PATHS),,$(PATH))' % main_output)
            for input in inputs:
                if not input.startswith('$(') and ' ' in input:
                    raise gyp.common.GypError('Action input filename "%s" in target %s contains a space' % (input, self.target))
            for output in outputs:
                if not output.startswith('$(') and ' ' in output:
                    raise gyp.common.GypError('Action output filename "%s" in target %s contains a space' % (output, self.target))
            self.WriteLn('%s: %s $(GYP_TARGET_DEPENDENCIES)' % (main_output, ' '.join(map(self.LocalPathify, inputs))))
            self.WriteLn('\t@echo "%s"' % quiet_cmd)
            self.WriteLn('\t$(hide)%s\n' % command)
            for output in outputs[1:]:
                self.WriteLn(f'{self.LocalPathify(output)}: {main_output} ;')
            extra_outputs += outputs
            self.WriteLn()
        self.WriteLn()

    def WriteRules(self, rules, extra_sources, extra_outputs):
        """Write Makefile code for any 'rules' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       rules (used to make other pieces dependent on these rules)
        """
        if len(rules) == 0:
            return
        for rule in rules:
            if len(rule.get('rule_sources', [])) == 0:
                continue
            name = make.StringToMakefileVariable('{}_{}'.format(self.relative_target, rule['rule_name']))
            self.WriteLn('\n### Generated for rule "%s":' % name)
            self.WriteLn('# "%s":' % rule)
            inputs = rule.get('inputs')
            for rule_source in rule.get('rule_sources', []):
                rule_source_dirname, rule_source_basename = os.path.split(rule_source)
                rule_source_root, rule_source_ext = os.path.splitext(rule_source_basename)
                outputs = [self.ExpandInputRoot(out, rule_source_root, rule_source_dirname) for out in rule['outputs']]
                dirs = set()
                for out in outputs:
                    if not out.startswith('$'):
                        print('WARNING: Rule for target %s writes output to local path %s' % (self.target, out))
                    dir = os.path.dirname(out)
                    if dir:
                        dirs.add(dir)
                extra_outputs += outputs
                if int(rule.get('process_outputs_as_sources', False)):
                    extra_sources.extend(outputs)
                components = []
                for component in rule['action']:
                    component = self.ExpandInputRoot(component, rule_source_root, rule_source_dirname)
                    if '$(RULE_SOURCES)' in component:
                        component = component.replace('$(RULE_SOURCES)', rule_source)
                    components.append(component)
                command = gyp.common.EncodePOSIXShellList(components)
                cd_action = 'cd $(gyp_local_path)/%s; ' % self.path
                command = cd_action + command
                if dirs:
                    command = 'mkdir -p %s' % ' '.join(dirs) + '; ' + command
                outputs = map(self.LocalPathify, outputs)
                main_output = outputs[0]
                self.WriteLn('%s: gyp_local_path := $(LOCAL_PATH)' % main_output)
                self.WriteLn('%s: gyp_var_prefix := $(GYP_VAR_PREFIX)' % main_output)
                self.WriteLn('%s: gyp_intermediate_dir := $(abspath $(gyp_intermediate_dir))' % main_output)
                self.WriteLn('%s: gyp_shared_intermediate_dir := $(abspath $(gyp_shared_intermediate_dir))' % main_output)
                self.WriteLn('%s: export PATH := $(subst $(ANDROID_BUILD_PATHS),,$(PATH))' % main_output)
                main_output_deps = self.LocalPathify(rule_source)
                if inputs:
                    main_output_deps += ' '
                    main_output_deps += ' '.join([self.LocalPathify(f) for f in inputs])
                self.WriteLn('%s: %s $(GYP_TARGET_DEPENDENCIES)' % (main_output, main_output_deps))
                self.WriteLn('\t%s\n' % command)
                for output in outputs[1:]:
                    self.WriteLn(f'{output}: {main_output} ;')
                self.WriteLn()
        self.WriteLn()

    def WriteCopies(self, copies, extra_outputs):
        """Write Makefile code for any 'copies' from the gyp input.

        extra_outputs: a list that will be filled in with any outputs of this action
                       (used to make other pieces dependent on this action)
        """
        self.WriteLn('### Generated for copy rule.')
        variable = make.StringToMakefileVariable(self.relative_target + '_copies')
        outputs = []
        for copy in copies:
            for path in copy['files']:
                if not copy['destination'].startswith('$'):
                    print('WARNING: Copy rule for target %s writes output to local path %s' % (self.target, copy['destination']))
                path = Sourceify(self.LocalPathify(path))
                filename = os.path.split(path)[1]
                output = Sourceify(self.LocalPathify(os.path.join(copy['destination'], filename)))
                self.WriteLn(f'{output}: {path} $(GYP_TARGET_DEPENDENCIES) | $(ACP)')
                self.WriteLn('\t@echo Copying: $@')
                self.WriteLn('\t$(hide) mkdir -p $(dir $@)')
                self.WriteLn('\t$(hide) $(ACP) -rpf $< $@')
                self.WriteLn()
                outputs.append(output)
        self.WriteLn('{} = {}'.format(variable, ' '.join(map(make.QuoteSpaces, outputs))))
        extra_outputs.append('$(%s)' % variable)
        self.WriteLn()

    def WriteSourceFlags(self, spec, configs):
        """Write out the flags and include paths used to compile source files for
        the current target.

        Args:
          spec, configs: input from gyp.
        """
        for configname, config in sorted(configs.items()):
            extracted_includes = []
            self.WriteLn('\n# Flags passed to both C and C++ files.')
            cflags, includes_from_cflags = self.ExtractIncludesFromCFlags(config.get('cflags', []) + config.get('cflags_c', []))
            extracted_includes.extend(includes_from_cflags)
            self.WriteList(cflags, 'MY_CFLAGS_%s' % configname)
            self.WriteList(config.get('defines'), 'MY_DEFS_%s' % configname, prefix='-D', quoter=make.EscapeCppDefine)
            self.WriteLn('\n# Include paths placed before CFLAGS/CPPFLAGS')
            includes = list(config.get('include_dirs', []))
            includes.extend(extracted_includes)
            includes = map(Sourceify, map(self.LocalPathify, includes))
            includes = self.NormalizeIncludePaths(includes)
            self.WriteList(includes, 'LOCAL_C_INCLUDES_%s' % configname)
            self.WriteLn('\n# Flags passed to only C++ (and not C) files.')
            self.WriteList(config.get('cflags_cc'), 'LOCAL_CPPFLAGS_%s' % configname)
        self.WriteLn('\nLOCAL_CFLAGS := $(MY_CFLAGS_$(GYP_CONFIGURATION)) $(MY_DEFS_$(GYP_CONFIGURATION))')
        if self.toolset == 'host':
            self.WriteLn('# Undefine ANDROID for host modules')
            self.WriteLn('LOCAL_CFLAGS += -UANDROID')
        self.WriteLn('LOCAL_C_INCLUDES := $(GYP_COPIED_SOURCE_ORIGIN_DIRS) $(LOCAL_C_INCLUDES_$(GYP_CONFIGURATION))')
        self.WriteLn('LOCAL_CPPFLAGS := $(LOCAL_CPPFLAGS_$(GYP_CONFIGURATION))')
        self.WriteLn('LOCAL_ASFLAGS := $(LOCAL_CFLAGS)')

    def WriteSources(self, spec, configs, extra_sources):
        """Write Makefile code for any 'sources' from the gyp input.
        These are source files necessary to build the current target.
        We need to handle shared_intermediate directory source files as
        a special case by copying them to the intermediate directory and
        treating them as a generated sources. Otherwise the Android build
        rules won't pick them up.

        Args:
          spec, configs: input from gyp.
          extra_sources: Sources generated from Actions or Rules.
        """
        sources = filter(make.Compilable, spec.get('sources', []))
        generated_not_sources = [x for x in extra_sources if not make.Compilable(x)]
        extra_sources = filter(make.Compilable, extra_sources)
        all_sources = sources + extra_sources
        local_cpp_extension = '.cpp'
        for source in all_sources:
            root, ext = os.path.splitext(source)
            if IsCPPExtension(ext):
                local_cpp_extension = ext
                break
        if local_cpp_extension != '.cpp':
            self.WriteLn('LOCAL_CPP_EXTENSION := %s' % local_cpp_extension)
        local_files = []
        for source in sources:
            root, ext = os.path.splitext(source)
            if '$(gyp_shared_intermediate_dir)' in source:
                extra_sources.append(source)
            elif '$(gyp_intermediate_dir)' in source:
                extra_sources.append(source)
            elif IsCPPExtension(ext) and ext != local_cpp_extension:
                extra_sources.append(source)
            else:
                local_files.append(os.path.normpath(os.path.join(self.path, source)))
        final_generated_sources = []
        origin_src_dirs = []
        for source in extra_sources:
            local_file = source
            if '$(gyp_intermediate_dir)/' not in local_file:
                basename = os.path.basename(local_file)
                local_file = '$(gyp_intermediate_dir)/' + basename
            root, ext = os.path.splitext(local_file)
            if IsCPPExtension(ext) and ext != local_cpp_extension:
                local_file = root + local_cpp_extension
            if local_file != source:
                self.WriteLn(f'{local_file}: {self.LocalPathify(source)}')
                self.WriteLn('\tmkdir -p $(@D); cp $< $@')
                origin_src_dirs.append(os.path.dirname(source))
            final_generated_sources.append(local_file)
        final_generated_sources.extend(generated_not_sources)
        self.WriteList(final_generated_sources, 'LOCAL_GENERATED_SOURCES')
        origin_src_dirs = gyp.common.uniquer(origin_src_dirs)
        origin_src_dirs = map(Sourceify, map(self.LocalPathify, origin_src_dirs))
        self.WriteList(origin_src_dirs, 'GYP_COPIED_SOURCE_ORIGIN_DIRS')
        self.WriteList(local_files, 'LOCAL_SRC_FILES')
        self.WriteSourceFlags(spec, configs)

    def ComputeAndroidModule(self, spec):
        """Return the Android module name used for a gyp spec.

        We use the complete qualified target name to avoid collisions between
        duplicate targets in different directories. We also add a suffix to
        distinguish gyp-generated module names.
        """
        if int(spec.get('android_unmangled_name', 0)):
            assert self.type != 'shared_library' or self.target.startswith('lib')
            return self.target
        if self.type == 'shared_library':
            prefix = 'lib_'
        else:
            prefix = ''
        if spec['toolset'] == 'host':
            suffix = '_$(TARGET_$(GYP_VAR_PREFIX)ARCH)_host_gyp'
        else:
            suffix = '_gyp'
        if self.path:
            middle = make.StringToMakefileVariable(f'{self.path}_{self.target}')
        else:
            middle = make.StringToMakefileVariable(self.target)
        return ''.join([prefix, middle, suffix])

    def ComputeOutputParts(self, spec):
        """Return the 'output basename' of a gyp spec, split into filename + ext.

        Android libraries must be named the same thing as their module name,
        otherwise the linker can't find them, so product_name and so on must be
        ignored if we are building a library, and the "lib" prepending is
        not done for Android.
        """
        assert self.type != 'loadable_module'
        target = spec['target_name']
        target_prefix = ''
        target_ext = ''
        if self.type == 'static_library':
            target = self.ComputeAndroidModule(spec)
            target_ext = '.a'
        elif self.type == 'shared_library':
            target = self.ComputeAndroidModule(spec)
            target_ext = '.so'
        elif self.type == 'none':
            target_ext = '.stamp'
        elif self.type != 'executable':
            print('ERROR: What output file should be generated?', 'type', self.type, 'target', target)
        if self.type != 'static_library' and self.type != 'shared_library':
            target_prefix = spec.get('product_prefix', target_prefix)
            target = spec.get('product_name', target)
            product_ext = spec.get('product_extension')
            if product_ext:
                target_ext = '.' + product_ext
        target_stem = target_prefix + target
        return (target_stem, target_ext)

    def ComputeOutputBasename(self, spec):
        """Return the 'output basename' of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          'libfoobar.so'
        """
        return ''.join(self.ComputeOutputParts(spec))

    def ComputeOutput(self, spec):
        """Return the 'output' (full output path) of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          '$(obj)/baz/libfoobar.so'
        """
        if self.type == 'executable':
            path = '$(gyp_shared_intermediate_dir)'
        elif self.type == 'shared_library':
            if self.toolset == 'host':
                path = '$($(GYP_HOST_VAR_PREFIX)HOST_OUT_INTERMEDIATE_LIBRARIES)'
            else:
                path = '$($(GYP_VAR_PREFIX)TARGET_OUT_INTERMEDIATE_LIBRARIES)'
        elif self.toolset == 'host':
            path = '$(call intermediates-dir-for,%s,%s,true,,$(GYP_HOST_VAR_PREFIX))' % (self.android_class, self.android_module)
        else:
            path = '$(call intermediates-dir-for,{},{},,,$(GYP_VAR_PREFIX))'.format(self.android_class, self.android_module)
        assert spec.get('product_dir') is None
        return os.path.join(path, self.ComputeOutputBasename(spec))

    def NormalizeIncludePaths(self, include_paths):
        """Normalize include_paths.
        Convert absolute paths to relative to the Android top directory.

        Args:
          include_paths: A list of unprocessed include paths.
        Returns:
          A list of normalized include paths.
        """
        normalized = []
        for path in include_paths:
            if path[0] == '/':
                path = gyp.common.RelativePath(path, self.android_top_dir)
            normalized.append(path)
        return normalized

    def ExtractIncludesFromCFlags(self, cflags):
        """Extract includes "-I..." out from cflags

        Args:
          cflags: A list of compiler flags, which may be mixed with "-I.."
        Returns:
          A tuple of lists: (clean_clfags, include_paths). "-I.." is trimmed.
        """
        clean_cflags = []
        include_paths = []
        for flag in cflags:
            if flag.startswith('-I'):
                include_paths.append(flag[2:])
            else:
                clean_cflags.append(flag)
        return (clean_cflags, include_paths)

    def FilterLibraries(self, libraries):
        """Filter the 'libraries' key to separate things that shouldn't be ldflags.

        Library entries that look like filenames should be converted to android
        module names instead of being passed to the linker as flags.

        Args:
          libraries: the value of spec.get('libraries')
        Returns:
          A tuple (static_lib_modules, dynamic_lib_modules, ldflags)
        """
        static_lib_modules = []
        dynamic_lib_modules = []
        ldflags = []
        for libs in libraries:
            for lib in libs.split():
                if lib == '-lc' or lib == '-lstdc++' or lib == '-lm' or lib.endswith('libgcc.a'):
                    continue
                match = re.search('([^/]+)\\.a$', lib)
                if match:
                    static_lib_modules.append(match.group(1))
                    continue
                match = re.search('([^/]+)\\.so$', lib)
                if match:
                    dynamic_lib_modules.append(match.group(1))
                    continue
                if lib.startswith('-l'):
                    ldflags.append(lib)
        return (static_lib_modules, dynamic_lib_modules, ldflags)

    def ComputeDeps(self, spec):
        """Compute the dependencies of a gyp spec.

        Returns a tuple (deps, link_deps), where each is a list of
        filenames that will need to be put in front of make for either
        building (deps) or linking (link_deps).
        """
        deps = []
        link_deps = []
        if 'dependencies' in spec:
            deps.extend([target_outputs[dep] for dep in spec['dependencies'] if target_outputs[dep]])
            for dep in spec['dependencies']:
                if dep in target_link_deps:
                    link_deps.append(target_link_deps[dep])
            deps.extend(link_deps)
        return (gyp.common.uniquer(deps), gyp.common.uniquer(link_deps))

    def WriteTargetFlags(self, spec, configs, link_deps):
        """Write Makefile code to specify the link flags and library dependencies.

        spec, configs: input from gyp.
        link_deps: link dependency list; see ComputeDeps()
        """
        libraries = gyp.common.uniquer(spec.get('libraries', []))
        static_libs, dynamic_libs, ldflags_libs = self.FilterLibraries(libraries)
        if self.type != 'static_library':
            for configname, config in sorted(configs.items()):
                ldflags = list(config.get('ldflags', []))
                self.WriteLn('')
                self.WriteList(ldflags, 'LOCAL_LDFLAGS_%s' % configname)
            self.WriteList(ldflags_libs, 'LOCAL_GYP_LIBS')
            self.WriteLn('LOCAL_LDFLAGS := $(LOCAL_LDFLAGS_$(GYP_CONFIGURATION)) $(LOCAL_GYP_LIBS)')
        if self.type != 'static_library':
            static_link_deps = [x[1] for x in link_deps if x[0] == 'static']
            shared_link_deps = [x[1] for x in link_deps if x[0] == 'shared']
        else:
            static_link_deps = []
            shared_link_deps = []
        if static_libs or static_link_deps:
            self.WriteLn('')
            self.WriteList(static_libs + static_link_deps, 'LOCAL_STATIC_LIBRARIES')
            self.WriteLn('# Enable grouping to fix circular references')
            self.WriteLn('LOCAL_GROUP_STATIC_LIBRARIES := true')
        if dynamic_libs or shared_link_deps:
            self.WriteLn('')
            self.WriteList(dynamic_libs + shared_link_deps, 'LOCAL_SHARED_LIBRARIES')

    def WriteTarget(self, spec, configs, deps, link_deps, part_of_all, write_alias_target):
        """Write Makefile code to produce the final target of the gyp spec.

        spec, configs: input from gyp.
        deps, link_deps: dependency lists; see ComputeDeps()
        part_of_all: flag indicating this target is part of 'all'
        write_alias_target: flag indicating whether to create short aliases for this
                            target
        """
        self.WriteLn('### Rules for final target.')
        if self.type != 'none':
            self.WriteTargetFlags(spec, configs, link_deps)
        settings = spec.get('aosp_build_settings', {})
        if settings:
            self.WriteLn('### Set directly by aosp_build_settings.')
            for k, v in settings.items():
                if isinstance(v, list):
                    self.WriteList(v, k)
                else:
                    self.WriteLn(f'{k} := {make.QuoteIfNecessary(v)}')
            self.WriteLn('')
        if part_of_all and write_alias_target:
            self.WriteLn('# Add target alias to "gyp_all_modules" target.')
            self.WriteLn('.PHONY: gyp_all_modules')
            self.WriteLn('gyp_all_modules: %s' % self.android_module)
            self.WriteLn('')
        if self.target != self.android_module and write_alias_target:
            self.WriteLn('# Alias gyp target name.')
            self.WriteLn('.PHONY: %s' % self.target)
            self.WriteLn(f'{self.target}: {self.android_module}')
            self.WriteLn('')
        modifier = ''
        if self.toolset == 'host':
            modifier = 'HOST_'
        if self.type == 'static_library':
            self.WriteLn('include $(BUILD_%sSTATIC_LIBRARY)' % modifier)
        elif self.type == 'shared_library':
            self.WriteLn('LOCAL_PRELINK_MODULE := false')
            self.WriteLn('include $(BUILD_%sSHARED_LIBRARY)' % modifier)
        elif self.type == 'executable':
            self.WriteLn('LOCAL_CXX_STL := libc++_static')
            self.WriteLn('LOCAL_MODULE_PATH := $(gyp_shared_intermediate_dir)')
            self.WriteLn('include $(BUILD_%sEXECUTABLE)' % modifier)
        else:
            self.WriteLn('LOCAL_MODULE_PATH := $(PRODUCT_OUT)/gyp_stamp')
            self.WriteLn('LOCAL_UNINSTALLABLE_MODULE := true')
            if self.toolset == 'target':
                self.WriteLn('LOCAL_2ND_ARCH_VAR_PREFIX := $(GYP_VAR_PREFIX)')
            else:
                self.WriteLn('LOCAL_2ND_ARCH_VAR_PREFIX := $(GYP_HOST_VAR_PREFIX)')
            self.WriteLn()
            self.WriteLn('include $(BUILD_SYSTEM)/base_rules.mk')
            self.WriteLn()
            self.WriteLn('$(LOCAL_BUILT_MODULE): $(LOCAL_ADDITIONAL_DEPENDENCIES)')
            self.WriteLn('\t$(hide) echo "Gyp timestamp: $@"')
            self.WriteLn('\t$(hide) mkdir -p $(dir $@)')
            self.WriteLn('\t$(hide) touch $@')
            self.WriteLn()
            self.WriteLn('LOCAL_2ND_ARCH_VAR_PREFIX :=')

    def WriteList(self, value_list, variable=None, prefix='', quoter=make.QuoteIfNecessary, local_pathify=False):
        """Write a variable definition that is a list of values.

        E.g. WriteList(['a','b'], 'foo', prefix='blah') writes out
             foo = blaha blahb
        but in a pretty-printed style.
        """
        values = ''
        if value_list:
            value_list = [quoter(prefix + value) for value in value_list]
            if local_pathify:
                value_list = [self.LocalPathify(value) for value in value_list]
            values = ' \\\n\t' + ' \\\n\t'.join(value_list)
        self.fp.write(f'{variable} :={values}\n\n')

    def WriteLn(self, text=''):
        self.fp.write(text + '\n')

    def LocalPathify(self, path):
        """Convert a subdirectory-relative path into a normalized path which starts
        with the make variable $(LOCAL_PATH) (i.e. the top of the project tree).
        Absolute paths, or paths that contain variables, are just normalized."""
        if '$(' in path or os.path.isabs(path):
            return os.path.normpath(path)
        local_path = os.path.join('$(LOCAL_PATH)', self.path, path)
        local_path = os.path.normpath(local_path)
        assert local_path.startswith('$(LOCAL_PATH)'), f'Path {path} attempts to escape from gyp path {self.path} !)'
        return local_path

    def ExpandInputRoot(self, template, expansion, dirname):
        if '%(INPUT_ROOT)s' not in template and '%(INPUT_DIRNAME)s' not in template:
            return template
        path = template % {'INPUT_ROOT': expansion, 'INPUT_DIRNAME': dirname}
        return os.path.normpath(path)