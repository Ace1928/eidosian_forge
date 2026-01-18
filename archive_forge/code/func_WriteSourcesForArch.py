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