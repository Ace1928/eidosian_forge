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