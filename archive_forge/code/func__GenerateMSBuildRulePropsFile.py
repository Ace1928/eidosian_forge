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
def _GenerateMSBuildRulePropsFile(props_path, msbuild_rules):
    """Generate the .props file."""
    content = ['Project', {'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003'}]
    for rule in msbuild_rules:
        content.extend([['PropertyGroup', {'Condition': "'$(%s)' == '' and '$(%s)' == '' and '$(ConfigurationType)' != 'Makefile'" % (rule.before_targets, rule.after_targets)}, [rule.before_targets, 'Midl'], [rule.after_targets, 'CustomBuild']], ['PropertyGroup', [rule.depends_on, {'Condition': "'$(ConfigurationType)' != 'Makefile'"}, '_SelectedFiles;$(%s)' % rule.depends_on]], ['ItemDefinitionGroup', [rule.rule_name, ['CommandLineTemplate', rule.command], ['Outputs', rule.outputs], ['ExecutionDescription', rule.description], ['AdditionalDependencies', rule.additional_dependencies]]]])
    easy_xml.WriteXmlIfChanged(content, props_path, pretty=True, win32=True)