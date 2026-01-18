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
def DisableForSourceTree(source_tree):
    for source in source_tree:
        if isinstance(source, MSVSProject.Filter):
            DisableForSourceTree(source.contents)
        else:
            basename, extension = os.path.splitext(source)
            if extension in extensions_excluded_from_precompile:
                for config_name, config in spec['configurations'].items():
                    tool = MSVSProject.Tool('VCCLCompilerTool', {'UsePrecompiledHeader': '0', 'ForcedIncludeFiles': '$(NOINHERIT)'})
                    p.AddFileConfig(_FixPath(source), _ConfigFullName(config_name, config), {}, tools=[tool])