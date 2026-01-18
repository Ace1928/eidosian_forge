import ntpath
import os
import posixpath
import re
import subprocess
import sys
from collections import OrderedDict
import gyp.common
import gyp.easy_xml as easy_xml
import gyp.generator.ninja as ninja_generator
import gyp.MSVSNew as MSVSNew
import gyp.MSVSProject as MSVSProject
import gyp.MSVSSettings as MSVSSettings
import gyp.MSVSToolFile as MSVSToolFile
import gyp.MSVSUserFile as MSVSUserFile
import gyp.MSVSUtil as MSVSUtil
import gyp.MSVSVersion as MSVSVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def _GetMSBuildProjectReferences(project):
    references = []
    if project.dependencies:
        group = ['ItemGroup']
        added_dependency_set = set()
        for dependency in project.dependencies:
            dependency_spec = dependency.spec
            should_skip_dep = False
            if project.spec['toolset'] == 'target':
                if dependency_spec['toolset'] == 'host':
                    if dependency_spec['type'] == 'static_library':
                        should_skip_dep = True
            if dependency.name.startswith('run_'):
                should_skip_dep = False
            if should_skip_dep:
                continue
            canonical_name = dependency.name.replace('_host', '')
            added_dependency_set.add(canonical_name)
            guid = dependency.guid
            project_dir = os.path.split(project.path)[0]
            relative_path = gyp.common.RelativePath(dependency.path, project_dir)
            project_ref = ['ProjectReference', {'Include': relative_path}, ['Project', guid], ['ReferenceOutputAssembly', 'false']]
            for config in dependency.spec.get('configurations', {}).values():
                if config.get('msvs_use_library_dependency_inputs', 0):
                    project_ref.append(['UseLibraryDependencyInputs', 'true'])
                    break
                if config.get('msvs_2010_disable_uldi_when_referenced', 0):
                    project_ref.append(['UseLibraryDependencyInputs', 'false'])
                    break
            group.append(project_ref)
        references.append(group)
    return references