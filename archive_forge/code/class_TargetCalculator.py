import gyp.common
import json
import os
import posixpath
class TargetCalculator:
    """Calculates the matching test_targets and matching compile_targets."""

    def __init__(self, files, additional_compile_target_names, test_target_names, data, target_list, target_dicts, toplevel_dir, build_files):
        self._additional_compile_target_names = set(additional_compile_target_names)
        self._test_target_names = set(test_target_names)
        self._name_to_target, self._changed_targets, self._root_targets = _GenerateTargets(data, target_list, target_dicts, toplevel_dir, frozenset(files), build_files)
        self._unqualified_mapping, self.invalid_targets = _GetUnqualifiedToTargetMapping(self._name_to_target, self._supplied_target_names_no_all())

    def _supplied_target_names(self):
        return self._additional_compile_target_names | self._test_target_names

    def _supplied_target_names_no_all(self):
        """Returns the supplied test targets without 'all'."""
        result = self._supplied_target_names()
        result.discard('all')
        return result

    def is_build_impacted(self):
        """Returns true if the supplied files impact the build at all."""
        return self._changed_targets

    def find_matching_test_target_names(self):
        """Returns the set of output test targets."""
        assert self.is_build_impacted()
        test_target_names_no_all = set(self._test_target_names)
        test_target_names_no_all.discard('all')
        test_targets_no_all = _LookupTargets(test_target_names_no_all, self._unqualified_mapping)
        test_target_names_contains_all = 'all' in self._test_target_names
        if test_target_names_contains_all:
            test_targets = [x for x in set(test_targets_no_all) | set(self._root_targets)]
        else:
            test_targets = [x for x in test_targets_no_all]
        print('supplied test_targets')
        for target_name in self._test_target_names:
            print('\t', target_name)
        print('found test_targets')
        for target in test_targets:
            print('\t', target.name)
        print('searching for matching test targets')
        matching_test_targets = _GetTargetsDependingOnMatchingTargets(test_targets)
        matching_test_targets_contains_all = test_target_names_contains_all and set(matching_test_targets) & set(self._root_targets)
        if matching_test_targets_contains_all:
            matching_test_targets = [x for x in set(matching_test_targets) & set(test_targets_no_all)]
        print('matched test_targets')
        for target in matching_test_targets:
            print('\t', target.name)
        matching_target_names = [gyp.common.ParseQualifiedTarget(target.name)[1] for target in matching_test_targets]
        if matching_test_targets_contains_all:
            matching_target_names.append('all')
            print('\tall')
        return matching_target_names

    def find_matching_compile_target_names(self):
        """Returns the set of output compile targets."""
        assert self.is_build_impacted()
        for target in self._name_to_target.values():
            target.visited = False
        supplied_targets = _LookupTargets(self._supplied_target_names_no_all(), self._unqualified_mapping)
        if 'all' in self._supplied_target_names():
            supplied_targets = [x for x in set(supplied_targets) | set(self._root_targets)]
        print('Supplied test_targets & compile_targets')
        for target in supplied_targets:
            print('\t', target.name)
        print('Finding compile targets')
        compile_targets = _GetCompileTargets(self._changed_targets, supplied_targets)
        return [gyp.common.ParseQualifiedTarget(target.name)[1] for target in compile_targets]