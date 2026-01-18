import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
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