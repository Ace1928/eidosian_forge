import gyp.common
import json
import os
import posixpath
def _GenerateTargets(data, target_list, target_dicts, toplevel_dir, files, build_files):
    """Returns a tuple of the following:
  . A dictionary mapping from fully qualified name to Target.
  . A list of the targets that have a source file in |files|.
  . Targets that constitute the 'all' target. See description at top of file
    for details on the 'all' target.
  This sets the |match_status| of the targets that contain any of the source
  files in |files| to MATCH_STATUS_MATCHES.
  |toplevel_dir| is the root of the source tree."""
    name_to_target = {}
    matching_targets = []
    targets_to_visit = target_list[:]
    build_file_in_files = {}
    roots = set()
    build_file_targets = set()
    while len(targets_to_visit) > 0:
        target_name = targets_to_visit.pop()
        created_target, target = _GetOrCreateTargetByName(name_to_target, target_name)
        if created_target:
            roots.add(target)
        elif target.visited:
            continue
        target.visited = True
        target.requires_build = _DoesTargetTypeRequireBuild(target_dicts[target_name])
        target_type = target_dicts[target_name]['type']
        target.is_executable = target_type == 'executable'
        target.is_static_library = target_type == 'static_library'
        target.is_or_has_linked_ancestor = target_type == 'executable' or target_type == 'shared_library'
        build_file = gyp.common.ParseQualifiedTarget(target_name)[0]
        if build_file not in build_file_in_files:
            build_file_in_files[build_file] = _WasBuildFileModified(build_file, data, files, toplevel_dir)
        if build_file in build_files:
            build_file_targets.add(target)
        if build_file_in_files[build_file]:
            print('matching target from modified build file', target_name)
            target.match_status = MATCH_STATUS_MATCHES
            matching_targets.append(target)
        else:
            sources = _ExtractSources(target_name, target_dicts[target_name], toplevel_dir)
            for source in sources:
                if _ToGypPath(os.path.normpath(source)) in files:
                    print('target', target_name, 'matches', source)
                    target.match_status = MATCH_STATUS_MATCHES
                    matching_targets.append(target)
                    break
        for dep in target_dicts[target_name].get('dependencies', []):
            targets_to_visit.append(dep)
            created_dep_target, dep_target = _GetOrCreateTargetByName(name_to_target, dep)
            if not created_dep_target:
                roots.discard(dep_target)
            target.deps.add(dep_target)
            dep_target.back_deps.add(target)
    return (name_to_target, matching_targets, roots & build_file_targets)