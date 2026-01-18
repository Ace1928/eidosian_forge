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
def _GenerateRulesForMSVS(p, output_dir, options, spec, sources, excluded_sources, actions_to_add):
    """Generate all the rules for a particular project.

  Arguments:
    p: the project
    output_dir: directory to emit rules to
    options: global options passed to the generator
    spec: the specification for this project
    sources: the set of all known source files in this project
    excluded_sources: the set of sources excluded from normal processing
    actions_to_add: deferred list of actions to add in
  """
    rules = spec.get('rules', [])
    rules_native = [r for r in rules if not int(r.get('msvs_external_rule', 0))]
    rules_external = [r for r in rules if int(r.get('msvs_external_rule', 0))]
    if rules_native:
        _GenerateNativeRulesForMSVS(p, rules_native, output_dir, spec, options)
    if rules_external:
        _GenerateExternalRules(rules_external, output_dir, spec, sources, options, actions_to_add)
    _AdjustSourcesForRules(rules, sources, excluded_sources, False)