import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteSources(self, configs, deps, sources, extra_outputs, extra_link_deps, part_of_all, precompiled_header):
    """Write Makefile code for any 'sources' from the gyp input.
        These are source files necessary to build the current target.

        configs, deps, sources: input from gyp.
        extra_outputs: a list of extra outputs this action should be dependent on;
                       used to serialize action/rules before compilation
        extra_link_deps: a list that will be filled in with any outputs of
                         compilation (to be used in link lines)
        part_of_all: flag indicating this target is part of 'all'
        """
    for configname in sorted(configs.keys()):
        config = configs[configname]
        self.WriteList(config.get('defines'), 'DEFS_%s' % configname, prefix='-D', quoter=EscapeCppDefine)
        if self.flavor == 'mac':
            cflags = self.xcode_settings.GetCflags(configname, arch=config.get('xcode_configuration_platform'))
            cflags_c = self.xcode_settings.GetCflagsC(configname)
            cflags_cc = self.xcode_settings.GetCflagsCC(configname)
            cflags_objc = self.xcode_settings.GetCflagsObjC(configname)
            cflags_objcc = self.xcode_settings.GetCflagsObjCC(configname)
        else:
            cflags = config.get('cflags')
            cflags_c = config.get('cflags_c')
            cflags_cc = config.get('cflags_cc')
        self.WriteLn('# Flags passed to all source files.')
        self.WriteList(cflags, 'CFLAGS_%s' % configname)
        self.WriteLn('# Flags passed to only C files.')
        self.WriteList(cflags_c, 'CFLAGS_C_%s' % configname)
        self.WriteLn('# Flags passed to only C++ files.')
        self.WriteList(cflags_cc, 'CFLAGS_CC_%s' % configname)
        if self.flavor == 'mac':
            self.WriteLn('# Flags passed to only ObjC files.')
            self.WriteList(cflags_objc, 'CFLAGS_OBJC_%s' % configname)
            self.WriteLn('# Flags passed to only ObjC++ files.')
            self.WriteList(cflags_objcc, 'CFLAGS_OBJCC_%s' % configname)
        includes = config.get('include_dirs')
        if includes:
            includes = [Sourceify(self.Absolutify(i)) for i in includes]
        self.WriteList(includes, 'INCS_%s' % configname, prefix='-I')
    compilable = list(filter(Compilable, sources))
    objs = [self.Objectify(self.Absolutify(Target(c))) for c in compilable]
    self.WriteList(objs, 'OBJS')
    for obj in objs:
        assert ' ' not in obj, 'Spaces in object filenames not supported (%s)' % obj
    self.WriteLn('# Add to the list of files we specially track dependencies for.')
    self.WriteLn('all_deps += $(OBJS)')
    self.WriteLn()
    if deps:
        self.WriteMakeRule(['$(OBJS)'], deps, comment='Make sure our dependencies are built before any of us.', order_only=True)
    if extra_outputs:
        self.WriteMakeRule(['$(OBJS)'], extra_outputs, comment='Make sure our actions/rules run before any of us.', order_only=True)
    pchdeps = precompiled_header.GetObjDependencies(compilable, objs)
    if pchdeps:
        self.WriteLn('# Dependencies from obj files to their precompiled headers')
        for source, obj, gch in pchdeps:
            self.WriteLn(f'{obj}: {gch}')
        self.WriteLn('# End precompiled header dependencies')
    if objs:
        extra_link_deps.append('$(OBJS)')
        self.WriteLn('# CFLAGS et al overrides must be target-local.\n# See "Target-specific Variable Values" in the GNU Make manual.')
        self.WriteLn('$(OBJS): TOOLSET := $(TOOLSET)')
        self.WriteLn('$(OBJS): GYP_CFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('c') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE))')
        self.WriteLn('$(OBJS): GYP_CXXFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('cc') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_CC_$(BUILDTYPE))')
        if self.flavor == 'mac':
            self.WriteLn('$(OBJS): GYP_OBJCFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('m') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE)) $(CFLAGS_OBJC_$(BUILDTYPE))')
            self.WriteLn('$(OBJS): GYP_OBJCXXFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) %s ' % precompiled_header.GetInclude('mm') + '$(CFLAGS_$(BUILDTYPE)) $(CFLAGS_CC_$(BUILDTYPE)) $(CFLAGS_OBJCC_$(BUILDTYPE))')
    self.WritePchTargets(precompiled_header.GetPchBuildCommands())
    extra_link_deps += [source for source in sources if Linkable(source)]
    self.WriteLn()