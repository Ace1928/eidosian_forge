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
def _BuildCommandLineForRule(spec, rule, has_input_path, do_setup_env):
    mcs = rule.get('msvs_cygwin_shell')
    if mcs is None:
        mcs = int(spec.get('msvs_cygwin_shell', 1))
    elif isinstance(mcs, str):
        mcs = int(mcs)
    quote_cmd = int(rule.get('msvs_quote_cmd', 1))
    return _BuildCommandLineForRuleRaw(spec, rule['action'], mcs, has_input_path, quote_cmd, do_setup_env=do_setup_env)