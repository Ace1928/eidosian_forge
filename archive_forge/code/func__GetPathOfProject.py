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
def _GetPathOfProject(qualified_target, spec, options, msvs_version):
    default_config = _GetDefaultConfiguration(spec)
    proj_filename = default_config.get('msvs_existing_vcproj')
    if not proj_filename:
        proj_filename = spec['target_name']
        if spec['toolset'] == 'host':
            proj_filename += '_host'
        proj_filename = proj_filename + options.suffix + msvs_version.ProjectExtension()
    build_file = gyp.common.BuildFile(qualified_target)
    proj_path = os.path.join(os.path.dirname(build_file), proj_filename)
    fix_prefix = None
    if options.generator_output:
        project_dir_path = os.path.dirname(os.path.abspath(proj_path))
        proj_path = os.path.join(options.generator_output, proj_path)
        fix_prefix = gyp.common.RelativePath(project_dir_path, os.path.dirname(proj_path))
    return (proj_path, fix_prefix)