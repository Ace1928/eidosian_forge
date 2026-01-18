import copy
import os
def InsertLargePdbShims(target_list, target_dicts, vars):
    """Insert a shim target that forces the linker to use 4KB pagesize PDBs.

  This is a workaround for targets with PDBs greater than 1GB in size, the
  limit for the 1KB pagesize PDBs created by the linker by default.

  Arguments:
    target_list: List of target pairs: 'base/base.gyp:base'.
    target_dicts: Dict of target properties keyed on target pair.
    vars: A dictionary of common GYP variables with generator-specific values.
  Returns:
    Tuple of the shimmed version of the inputs.
  """
    targets_to_shim = []
    for t in target_dicts:
        target_dict = target_dicts[t]
        if not int(target_dict.get('msvs_large_pdb', 0)):
            continue
        targets_to_shim.append(t)
    large_pdb_shim_cc = _GetLargePdbShimCcPath()
    for t in targets_to_shim:
        target_dict = target_dicts[t]
        target_name = target_dict.get('target_name')
        base_dict = _DeepCopySomeKeys(target_dict, ['configurations', 'default_configuration', 'toolset'])
        copy_suffix = 'large_pdb_copy'
        copy_target_name = target_name + '_' + copy_suffix
        full_copy_target_name = _SuffixName(t, copy_suffix)
        shim_cc_basename = os.path.basename(large_pdb_shim_cc)
        shim_cc_dir = vars['SHARED_INTERMEDIATE_DIR'] + '/' + copy_target_name
        shim_cc_path = shim_cc_dir + '/' + shim_cc_basename
        copy_dict = copy.deepcopy(base_dict)
        copy_dict['target_name'] = copy_target_name
        copy_dict['type'] = 'none'
        copy_dict['sources'] = [large_pdb_shim_cc]
        copy_dict['copies'] = [{'destination': shim_cc_dir, 'files': [large_pdb_shim_cc]}]
        shim_suffix = 'large_pdb_shim'
        shim_target_name = target_name + '_' + shim_suffix
        full_shim_target_name = _SuffixName(t, shim_suffix)
        shim_dict = copy.deepcopy(base_dict)
        shim_dict['target_name'] = shim_target_name
        shim_dict['type'] = 'static_library'
        shim_dict['sources'] = [shim_cc_path]
        shim_dict['dependencies'] = [full_copy_target_name]
        for config_name, config in shim_dict.get('configurations').items():
            pdb_path = _GetPdbPath(target_dict, config_name, vars)
            for key in ['msvs_precompiled_header', 'msvs_precompiled_source', 'test']:
                config.pop(key, None)
            msvs = config.setdefault('msvs_settings', {})
            compiler = msvs.setdefault('VCCLCompilerTool', {})
            compiler['DebugInformationFormat'] = '3'
            compiler['ProgramDataBaseFileName'] = pdb_path
            config = target_dict['configurations'][config_name]
            msvs = config.setdefault('msvs_settings', {})
            linker = msvs.setdefault('VCLinkerTool', {})
            linker['GenerateDebugInformation'] = 'true'
            linker['ProgramDatabaseFile'] = pdb_path
        target_list.insert(0, full_copy_target_name)
        target_list.insert(0, full_shim_target_name)
        target_dicts[full_copy_target_name] = copy_dict
        target_dicts[full_shim_target_name] = shim_dict
        target_dict.setdefault('dependencies', []).append(full_shim_target_name)
    return (target_list, target_dicts)