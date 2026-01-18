import gyp.common
import gyp.xcode_emulation
import json
import os
def AddCommandsForTarget(cwd, target, params, per_config_commands):
    output_dir = params['generator_flags'].get('output_dir', 'out')
    for configuration_name, configuration in target['configurations'].items():
        if IsMac(params):
            xcode_settings = gyp.xcode_emulation.XcodeSettings(target)
            cflags = xcode_settings.GetCflags(configuration_name)
            cflags_c = xcode_settings.GetCflagsC(configuration_name)
            cflags_cc = xcode_settings.GetCflagsCC(configuration_name)
        else:
            cflags = configuration.get('cflags', [])
            cflags_c = configuration.get('cflags_c', [])
            cflags_cc = configuration.get('cflags_cc', [])
        cflags_c = cflags + cflags_c
        cflags_cc = cflags + cflags_cc
        defines = configuration.get('defines', [])
        defines = ['-D' + s for s in defines]
        extensions = ('.c', '.cc', '.cpp', '.cxx')
        sources = [s for s in target.get('sources', []) if s.endswith(extensions)]

        def resolve(filename):
            return os.path.abspath(os.path.join(cwd, filename))
        include_dirs = configuration.get('include_dirs', [])
        include_dirs = [s for s in include_dirs if not s.startswith('$(obj)')]
        includes = ['-I' + resolve(s) for s in include_dirs]
        defines = gyp.common.EncodePOSIXShellList(defines)
        includes = gyp.common.EncodePOSIXShellList(includes)
        cflags_c = gyp.common.EncodePOSIXShellList(cflags_c)
        cflags_cc = gyp.common.EncodePOSIXShellList(cflags_cc)
        commands = per_config_commands.setdefault(configuration_name, [])
        for source in sources:
            file = resolve(source)
            isc = source.endswith('.c')
            cc = 'cc' if isc else 'c++'
            cflags = cflags_c if isc else cflags_cc
            command = ' '.join((cc, defines, includes, cflags, '-c', gyp.common.EncodePOSIXShellArgument(file)))
            commands.append(dict(command=command, directory=output_dir, file=file))