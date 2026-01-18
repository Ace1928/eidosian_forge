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
def _AdjustSourcesForRules(rules, sources, excluded_sources, is_msbuild):
    for rule in rules:
        trigger_files = _FindRuleTriggerFiles(rule, sources)
        for trigger_file in trigger_files:
            excluded_sources.discard(_FixPath(trigger_file))
            if int(rule.get('process_outputs_as_sources', False)):
                inputs, outputs = _RuleInputsAndOutputs(rule, trigger_file)
                inputs = OrderedSet(_FixPaths(inputs))
                outputs = OrderedSet(_FixPaths(outputs))
                inputs.remove(_FixPath(trigger_file))
                sources.update(inputs)
                if not is_msbuild:
                    excluded_sources.update(inputs)
                sources.update(outputs)