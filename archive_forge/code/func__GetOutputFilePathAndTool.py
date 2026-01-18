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
def _GetOutputFilePathAndTool(spec, msbuild):
    """Returns the path and tool to use for this target.

  Figures out the path of the file this spec will create and the name of
  the VC tool that will create it.

  Arguments:
    spec: The target dictionary containing the properties of the target.
  Returns:
    A triple of (file path, name of the vc tool, name of the msbuild tool)
  """
    out_file = ''
    vc_tool = ''
    msbuild_tool = ''
    output_file_map = {'executable': ('VCLinkerTool', 'Link', '$(OutDir)', '.exe'), 'shared_library': ('VCLinkerTool', 'Link', '$(OutDir)', '.dll'), 'loadable_module': ('VCLinkerTool', 'Link', '$(OutDir)', '.dll'), 'windows_driver': ('VCLinkerTool', 'Link', '$(OutDir)', '.sys'), 'static_library': ('VCLibrarianTool', 'Lib', '$(OutDir)lib\\', '.lib')}
    output_file_props = output_file_map.get(spec['type'])
    if output_file_props and int(spec.get('msvs_auto_output_file', 1)):
        vc_tool, msbuild_tool, out_dir, suffix = output_file_props
        if spec.get('standalone_static_library', 0):
            out_dir = '$(OutDir)'
        out_dir = spec.get('product_dir', out_dir)
        product_extension = spec.get('product_extension')
        if product_extension:
            suffix = '.' + product_extension
        elif msbuild:
            suffix = '$(TargetExt)'
        prefix = spec.get('product_prefix', '')
        product_name = spec.get('product_name', '$(ProjectName)')
        out_file = ntpath.join(out_dir, prefix + product_name + suffix)
    return (out_file, vc_tool, msbuild_tool)