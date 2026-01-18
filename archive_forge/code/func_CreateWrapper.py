import errno
import gyp.generator.ninja
import os
import re
import xml.sax.saxutils
def CreateWrapper(target_list, target_dicts, data, params):
    """Initialize targets for the ninja wrapper.

  This sets up the necessary variables in the targets to generate Xcode projects
  that use ninja as an external builder.
  Arguments:
    target_list: List of target pairs: 'base/base.gyp:base'.
    target_dicts: Dict of target properties keyed on target pair.
    data: Dict of flattened build files keyed on gyp path.
    params: Dict of global options for gyp.
  """
    orig_gyp = params['build_files'][0]
    for gyp_name, gyp_dict in data.items():
        if gyp_name == orig_gyp:
            depth = gyp_dict['_DEPTH']
    generator_flags = params.get('generator_flags', {})
    main_gyp = generator_flags.get('xcode_ninja_main_gyp', None)
    if main_gyp is None:
        build_file_root, build_file_ext = os.path.splitext(orig_gyp)
        main_gyp = build_file_root + '.ninja' + build_file_ext
    new_target_list = []
    new_target_dicts = {}
    new_data = {}
    new_data[main_gyp] = {}
    new_data[main_gyp]['included_files'] = []
    new_data[main_gyp]['targets'] = []
    new_data[main_gyp]['xcode_settings'] = data[orig_gyp].get('xcode_settings', {})
    executable_target_pattern = generator_flags.get('xcode_ninja_executable_target_pattern', None)
    target_extras = generator_flags.get('xcode_ninja_target_pattern', None)
    for old_qualified_target in target_list:
        spec = target_dicts[old_qualified_target]
        if IsValidTargetForWrapper(target_extras, executable_target_pattern, spec):
            target_name = spec.get('target_name')
            new_target_name = f'{main_gyp}:{target_name}#target'
            new_target_list.append(new_target_name)
            new_target_dicts[new_target_name] = _TargetFromSpec(spec, params)
            for old_target in data[old_qualified_target.split(':')[0]]['targets']:
                if old_target['target_name'] == target_name:
                    new_data_target = {}
                    new_data_target['target_name'] = old_target['target_name']
                    new_data_target['toolset'] = old_target['toolset']
                    new_data[main_gyp]['targets'].append(new_data_target)
    sources_target_name = 'sources_for_indexing'
    sources_target = _TargetFromSpec({'target_name': sources_target_name, 'toolset': 'target', 'default_configuration': 'Default', 'mac_bundle': '0', 'type': 'executable'}, None)
    sources_target['configurations'] = {'Default': {'include_dirs': [depth]}}
    skip_excluded_files = not generator_flags.get('xcode_ninja_list_excluded_files', True)
    sources = []
    for target, target_dict in target_dicts.items():
        base = os.path.dirname(target)
        files = target_dict.get('sources', []) + target_dict.get('mac_bundle_resources', [])
        if not skip_excluded_files:
            files.extend(target_dict.get('sources_excluded', []) + target_dict.get('mac_bundle_resources_excluded', []))
        for action in target_dict.get('actions', []):
            files.extend(action.get('inputs', []))
            if not skip_excluded_files:
                files.extend(action.get('inputs_excluded', []))
        files = [file for file in files if not file.startswith('$')]
        relative_path = os.path.dirname(main_gyp)
        sources += [os.path.relpath(os.path.join(base, file), relative_path) for file in files]
    sources_target['sources'] = sorted(set(sources))
    sources_gyp = os.path.join(os.path.dirname(main_gyp), sources_target_name + '.gyp')
    fully_qualified_target_name = f'{sources_gyp}:{sources_target_name}#target'
    new_target_list.append(fully_qualified_target_name)
    new_target_dicts[fully_qualified_target_name] = sources_target
    new_data_target = {}
    new_data_target['target_name'] = sources_target['target_name']
    new_data_target['_DEPTH'] = depth
    new_data_target['toolset'] = 'target'
    new_data[sources_gyp] = {}
    new_data[sources_gyp]['targets'] = []
    new_data[sources_gyp]['included_files'] = []
    new_data[sources_gyp]['xcode_settings'] = data[orig_gyp].get('xcode_settings', {})
    new_data[sources_gyp]['targets'].append(new_data_target)
    _WriteWorkspace(main_gyp, sources_gyp, params)
    return (new_target_list, new_target_dicts, new_data)