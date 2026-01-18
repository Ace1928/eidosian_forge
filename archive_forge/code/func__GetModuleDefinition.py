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
def _GetModuleDefinition(spec):
    def_file = ''
    if spec['type'] in ['shared_library', 'loadable_module', 'executable', 'windows_driver']:
        def_files = [s for s in spec.get('sources', []) if s.endswith('.def')]
        if len(def_files) == 1:
            def_file = _FixPath(def_files[0])
        elif def_files:
            raise ValueError('Multiple module definition files in one target, target %s lists multiple .def files: %s' % (spec['target_name'], ' '.join(def_files)))
    return def_file