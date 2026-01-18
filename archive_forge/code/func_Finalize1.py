import filecmp
import gyp.common
import gyp.xcodeproj_file
import gyp.xcode_ninja
import errno
import os
import sys
import posixpath
import re
import shutil
import subprocess
import tempfile
def Finalize1(self, xcode_targets, serialize_all_tests):
    configurations = []
    for xct in self.project.GetProperty('targets'):
        xccl = xct.GetProperty('buildConfigurationList')
        xcbcs = xccl.GetProperty('buildConfigurations')
        for xcbc in xcbcs:
            name = xcbc.GetProperty('name')
            if name not in configurations:
                configurations.append(name)
    try:
        xccl = CreateXCConfigurationList(configurations)
        self.project.SetProperty('buildConfigurationList', xccl)
    except Exception:
        sys.stderr.write('Problem with gyp file %s\n' % self.gyp_path)
        raise
    xccl.SetBuildSetting(_intermediate_var, '$(PROJECT_DERIVED_FILE_DIR)/$(CONFIGURATION)')
    xccl.SetBuildSetting(_shared_intermediate_var, '$(SYMROOT)/DerivedSources/$(CONFIGURATION)')
    for xck, xcv in self.build_file_dict.get('xcode_settings', {}).items():
        xccl.SetBuildSetting(xck, xcv)
    if 'xcode_config_file' in self.build_file_dict:
        config_ref = self.project.AddOrGetFileInRootGroup(self.build_file_dict['xcode_config_file'])
        xccl.SetBaseConfiguration(config_ref)
    build_file_configurations = self.build_file_dict.get('configurations', {})
    if build_file_configurations:
        for config_name in configurations:
            build_file_configuration_named = build_file_configurations.get(config_name, {})
            if build_file_configuration_named:
                xcc = xccl.ConfigurationNamed(config_name)
                for xck, xcv in build_file_configuration_named.get('xcode_settings', {}).items():
                    xcc.SetBuildSetting(xck, xcv)
                if 'xcode_config_file' in build_file_configuration_named:
                    config_ref = self.project.AddOrGetFileInRootGroup(build_file_configurations[config_name]['xcode_config_file'])
                    xcc.SetBaseConfiguration(config_ref)
    ordinary_targets = []
    run_test_targets = []
    support_targets = []
    targets = []
    has_custom_all = False
    targets_for_all = []
    for target in self.build_file_dict['targets']:
        target_name = target['target_name']
        toolset = target['toolset']
        qualified_target = gyp.common.QualifiedTarget(self.gyp_path, target_name, toolset)
        xcode_target = xcode_targets[qualified_target]
        assert xcode_target in self.project._properties['targets']
        targets.append(xcode_target)
        ordinary_targets.append(xcode_target)
        if xcode_target.support_target:
            support_targets.append(xcode_target.support_target)
            targets.append(xcode_target.support_target)
        if not int(target.get('suppress_wildcard', False)):
            targets_for_all.append(xcode_target)
        if target_name.lower() == 'all':
            has_custom_all = True
        if target.get('run_as'):
            xccl = CreateXCConfigurationList(configurations)
            run_target = gyp.xcodeproj_file.PBXAggregateTarget({'name': 'Run ' + target_name, 'productName': xcode_target.GetProperty('productName'), 'buildConfigurationList': xccl}, parent=self.project)
            run_target.AddDependency(xcode_target)
            command = target['run_as']
            script = ''
            if command.get('working_directory'):
                script = script + 'cd "%s"\n' % gyp.xcodeproj_file.ConvertVariablesToShellSyntax(command.get('working_directory'))
            if command.get('environment'):
                script = script + '\n'.join(['export %s="%s"' % (key, gyp.xcodeproj_file.ConvertVariablesToShellSyntax(val)) for key, val in command.get('environment').items()]) + '\n'
            command_prefix = ''
            if serialize_all_tests:
                command_prefix = 'python -c "import fcntl, subprocess, sys\nfile = open(\'$TMPDIR/GYP_serialize_test_runs\', \'a\')\nfcntl.flock(file.fileno(), fcntl.LOCK_EX)\nsys.exit(subprocess.call(sys.argv[1:]))" '
            script = script + 'exec ' + command_prefix + '%s\nexit 1\n' % gyp.xcodeproj_file.ConvertVariablesToShellSyntax(gyp.common.EncodePOSIXShellList(command.get('action')))
            ssbp = gyp.xcodeproj_file.PBXShellScriptBuildPhase({'shellScript': script, 'showEnvVarsInLog': 0})
            run_target.AppendProperty('buildPhases', ssbp)
            targets.append(run_target)
            run_test_targets.append(run_target)
            xcode_target.test_runner = run_target
    assert len(self.project._properties['targets']) == len(ordinary_targets) + len(support_targets)
    self.project._properties['targets'] = targets
    self.project.RootGroupsTakeOverOnlyChildren(True)
    self.project.SortGroups()
    if len(targets_for_all) > 1 and (not has_custom_all):
        xccl = CreateXCConfigurationList(configurations)
        all_target = gyp.xcodeproj_file.PBXAggregateTarget({'buildConfigurationList': xccl, 'name': 'All'}, parent=self.project)
        for target in targets_for_all:
            all_target.AddDependency(target)
        self.project._properties['targets'].insert(0, all_target)
    if len(run_test_targets) > 1:
        xccl = CreateXCConfigurationList(configurations)
        run_all_tests_target = gyp.xcodeproj_file.PBXAggregateTarget({'buildConfigurationList': xccl, 'name': 'Run All Tests'}, parent=self.project)
        for run_test_target in run_test_targets:
            run_all_tests_target.AddDependency(run_test_target)
        self.project._properties['targets'].insert(1, run_all_tests_target)