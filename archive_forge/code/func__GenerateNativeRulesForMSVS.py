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
def _GenerateNativeRulesForMSVS(p, rules, output_dir, spec, options):
    """Generate a native rules file.

  Arguments:
    p: the target project
    rules: the set of rules to include
    output_dir: the directory in which the project/gyp resides
    spec: the project dict
    options: global generator options
  """
    rules_filename = '{}{}.rules'.format(spec['target_name'], options.suffix)
    rules_file = MSVSToolFile.Writer(os.path.join(output_dir, rules_filename), spec['target_name'])
    for r in rules:
        rule_name = r['rule_name']
        rule_ext = r['extension']
        inputs = _FixPaths(r.get('inputs', []))
        outputs = _FixPaths(r.get('outputs', []))
        if 'action' not in r and (not r.get('rule_sources', [])):
            continue
        cmd = _BuildCommandLineForRule(spec, r, has_input_path=True, do_setup_env=True)
        rules_file.AddCustomBuildRule(name=rule_name, description=r.get('message', rule_name), extensions=[rule_ext], additional_dependencies=inputs, outputs=outputs, cmd=cmd)
    rules_file.WriteIfChanged()
    p.AddToolFile(rules_filename)