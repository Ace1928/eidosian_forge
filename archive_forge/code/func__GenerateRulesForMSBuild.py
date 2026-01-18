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
def _GenerateRulesForMSBuild(output_dir, options, spec, sources, excluded_sources, props_files_of_rules, targets_files_of_rules, actions_to_add, rule_dependencies, extension_to_rule_name):
    rules = spec.get('rules', [])
    rules_native = [r for r in rules if not int(r.get('msvs_external_rule', 0))]
    rules_external = [r for r in rules if int(r.get('msvs_external_rule', 0))]
    msbuild_rules = []
    for rule in rules_native:
        if 'action' not in rule and (not rule.get('rule_sources', [])):
            continue
        msbuild_rule = MSBuildRule(rule, spec)
        msbuild_rules.append(msbuild_rule)
        rule_dependencies.update(msbuild_rule.additional_dependencies.split(';'))
        extension_to_rule_name[msbuild_rule.extension] = msbuild_rule.rule_name
    if msbuild_rules:
        base = spec['target_name'] + options.suffix
        props_name = base + '.props'
        targets_name = base + '.targets'
        xml_name = base + '.xml'
        props_files_of_rules.add(props_name)
        targets_files_of_rules.add(targets_name)
        props_path = os.path.join(output_dir, props_name)
        targets_path = os.path.join(output_dir, targets_name)
        xml_path = os.path.join(output_dir, xml_name)
        _GenerateMSBuildRulePropsFile(props_path, msbuild_rules)
        _GenerateMSBuildRuleTargetsFile(targets_path, msbuild_rules)
        _GenerateMSBuildRuleXmlFile(xml_path, msbuild_rules)
    if rules_external:
        _GenerateExternalRules(rules_external, output_dir, spec, sources, options, actions_to_add)
    _AdjustSourcesForRules(rules, sources, excluded_sources, True)