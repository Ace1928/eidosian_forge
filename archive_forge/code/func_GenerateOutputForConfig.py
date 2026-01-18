import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def GenerateOutputForConfig(target_list, target_dicts, data, params, config_to_use):
    options = params['options']
    generator_flags = params['generator_flags']
    flavor = gyp.common.GetFlavor(params)
    generator_dir = os.path.relpath(options.generator_output or '.')
    output_dir = generator_flags.get('output_dir', 'out')
    build_dir = os.path.normpath(os.path.join(generator_dir, output_dir, config_to_use))
    toplevel_build = os.path.join(options.toplevel_dir, build_dir)
    output_file = os.path.join(toplevel_build, 'CMakeLists.txt')
    gyp.common.EnsureDirExists(output_file)
    output = open(output_file, 'w')
    output.write('cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)\n')
    output.write('cmake_policy(VERSION 2.8.8)\n')
    gyp_file, project_target, _ = gyp.common.ParseQualifiedTarget(target_list[-1])
    output.write('project(')
    output.write(project_target)
    output.write(')\n')
    SetVariable(output, 'configuration', config_to_use)
    ar = None
    cc = None
    cxx = None
    make_global_settings = data[gyp_file].get('make_global_settings', [])
    build_to_top = gyp.common.InvertRelativePath(build_dir, options.toplevel_dir)
    for key, value in make_global_settings:
        if key == 'AR':
            ar = os.path.join(build_to_top, value)
        if key == 'CC':
            cc = os.path.join(build_to_top, value)
        if key == 'CXX':
            cxx = os.path.join(build_to_top, value)
    ar = gyp.common.GetEnvironFallback(['AR_target', 'AR'], ar)
    cc = gyp.common.GetEnvironFallback(['CC_target', 'CC'], cc)
    cxx = gyp.common.GetEnvironFallback(['CXX_target', 'CXX'], cxx)
    if ar:
        SetVariable(output, 'CMAKE_AR', ar)
    if cc:
        SetVariable(output, 'CMAKE_C_COMPILER', cc)
    if cxx:
        SetVariable(output, 'CMAKE_CXX_COMPILER', cxx)
    output.write('enable_language(ASM)\n')
    if cc:
        SetVariable(output, 'CMAKE_ASM_COMPILER', cc)
    SetVariable(output, 'builddir', '${CMAKE_CURRENT_BINARY_DIR}')
    SetVariable(output, 'obj', '${builddir}/obj')
    output.write('\n')
    output.write('set(CMAKE_C_OUTPUT_EXTENSION_REPLACE 1)\n')
    output.write('set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)\n')
    output.write('\n')
    if flavor != 'mac':
        output.write('set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1)\n')
    output.write('\n')
    namer = CMakeNamer(target_list)
    all_qualified_targets = set()
    for build_file in params['build_files']:
        for qualified_target in gyp.common.AllTargets(target_list, target_dicts, os.path.normpath(build_file)):
            all_qualified_targets.add(qualified_target)
    for qualified_target in target_list:
        if flavor == 'mac':
            gyp_file, _, _ = gyp.common.ParseQualifiedTarget(qualified_target)
            spec = target_dicts[qualified_target]
            gyp.xcode_emulation.MergeGlobalXcodeSettingsToSpec(data[gyp_file], spec)
        WriteTarget(namer, qualified_target, target_dicts, build_dir, config_to_use, options, generator_flags, all_qualified_targets, flavor, output)
    output.close()