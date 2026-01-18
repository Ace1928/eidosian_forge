import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def ExpandRuleVariables(self, path, root, dirname, source, ext, name):
    if self.flavor == 'win':
        path = self.msvs_settings.ConvertVSMacros(path, config=self.config_name)
    path = path.replace(generator_default_variables['RULE_INPUT_ROOT'], root)
    path = path.replace(generator_default_variables['RULE_INPUT_DIRNAME'], dirname)
    path = path.replace(generator_default_variables['RULE_INPUT_PATH'], source)
    path = path.replace(generator_default_variables['RULE_INPUT_EXT'], ext)
    path = path.replace(generator_default_variables['RULE_INPUT_NAME'], name)
    return path