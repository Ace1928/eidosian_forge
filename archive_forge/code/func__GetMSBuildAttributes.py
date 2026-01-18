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
def _GetMSBuildAttributes(spec, config, build_file):
    if 'msbuild_configuration_attributes' not in config:
        msbuild_attributes = _ConvertMSVSBuildAttributes(spec, config, build_file)
    else:
        config_type = _GetMSVSConfigurationType(spec, build_file)
        config_type = _ConvertMSVSConfigurationType(config_type)
        msbuild_attributes = config.get('msbuild_configuration_attributes', {})
        msbuild_attributes.setdefault('ConfigurationType', config_type)
        output_dir = msbuild_attributes.get('OutputDirectory', '$(SolutionDir)$(Configuration)')
        msbuild_attributes['OutputDirectory'] = _FixPath(output_dir) + '\\'
        if 'IntermediateDirectory' not in msbuild_attributes:
            intermediate = _FixPath('$(Configuration)') + '\\'
            msbuild_attributes['IntermediateDirectory'] = intermediate
        if 'CharacterSet' in msbuild_attributes:
            msbuild_attributes['CharacterSet'] = _ConvertMSVSCharacterSet(msbuild_attributes['CharacterSet'])
    if 'TargetName' not in msbuild_attributes:
        prefix = spec.get('product_prefix', '')
        product_name = spec.get('product_name', '$(ProjectName)')
        target_name = prefix + product_name
        msbuild_attributes['TargetName'] = target_name
    if 'TargetExt' not in msbuild_attributes and 'product_extension' in spec:
        ext = spec.get('product_extension')
        msbuild_attributes['TargetExt'] = '.' + ext
    if spec.get('msvs_external_builder'):
        external_out_dir = spec.get('msvs_external_builder_out_dir', '.')
        msbuild_attributes['OutputDirectory'] = _FixPath(external_out_dir) + '\\'
    msbuild_tool_map = {'executable': 'Link', 'shared_library': 'Link', 'loadable_module': 'Link', 'windows_driver': 'Link', 'static_library': 'Lib'}
    msbuild_tool = msbuild_tool_map.get(spec['type'])
    if msbuild_tool:
        msbuild_settings = config['finalized_msbuild_settings']
        out_file = msbuild_settings[msbuild_tool].get('OutputFile')
        if out_file:
            msbuild_attributes['TargetPath'] = _FixPath(out_file)
        target_ext = msbuild_settings[msbuild_tool].get('TargetExt')
        if target_ext:
            msbuild_attributes['TargetExt'] = target_ext
    return msbuild_attributes