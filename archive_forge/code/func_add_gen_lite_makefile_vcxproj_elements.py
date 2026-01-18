from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def add_gen_lite_makefile_vcxproj_elements(self, root: ET.Element, platform: str, target_ext: str, vslite_ctx: dict, target, proj_to_build_root: str, primary_src_lang: T.Optional[str]) -> None:
    ET.SubElement(root, 'ImportGroup', Label='ExtensionSettings')
    ET.SubElement(root, 'ImportGroup', Label='Shared')
    prop_sheets_grp = ET.SubElement(root, 'ImportGroup', Label='PropertySheets')
    ET.SubElement(prop_sheets_grp, 'Import', {'Project': '$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props', 'Condition': "exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')", 'Label': 'LocalAppDataPlatform'})
    ET.SubElement(root, 'PropertyGroup', Label='UserMacros')
    nmake_base_meson_command, exe_search_paths = Vs2010Backend.get_nmake_base_meson_command_and_exe_search_paths()
    proj_to_multiconfigured_builds_parent_dir = os.path.join(proj_to_build_root, '..')
    multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list()
    for buildtype in multi_config_buildtype_list:
        per_config_prop_group = ET.SubElement(root, 'PropertyGroup', Condition=f"'$(Configuration)|$(Platform)'=='{buildtype}|{platform}'")
        _, build_dir_tail = os.path.split(self.src_to_build)
        meson_build_dir_for_buildtype = build_dir_tail[:-2] + buildtype
        proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
        ET.SubElement(per_config_prop_group, 'OutDir').text = f'{proj_to_build_dir_for_buildtype}\\'
        ET.SubElement(per_config_prop_group, 'IntDir').text = f'{proj_to_build_dir_for_buildtype}\\'
        ET.SubElement(per_config_prop_group, 'NMakeBuildCommandLine').text = f'{nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}"'
        ET.SubElement(per_config_prop_group, 'NMakeOutput').text = f'$(OutDir){target.name}{target_ext}'
        captured_build_args = vslite_ctx[buildtype][target.get_id()]
        ET.SubElement(per_config_prop_group, 'NMakeReBuildCommandLine').text = f'{nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}" --clean && {nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}"'
        ET.SubElement(per_config_prop_group, 'NMakeCleanCommandLine').text = f'{nmake_base_meson_command} compile -C "{proj_to_build_dir_for_buildtype}" --clean'
        ET.SubElement(per_config_prop_group, 'ExecutablePath').text = exe_search_paths
        if primary_src_lang:
            primary_src_type_build_args = captured_build_args[primary_src_lang]
            preproc_defs, inc_paths, other_compile_opts = Vs2010Backend._extract_nmake_fields(primary_src_type_build_args)
            ET.SubElement(per_config_prop_group, 'NMakePreprocessorDefinitions').text = preproc_defs
            ET.SubElement(per_config_prop_group, 'NMakeIncludeSearchPath').text = inc_paths
            ET.SubElement(per_config_prop_group, 'AdditionalOptions').text = other_compile_opts
        ET.SubElement(per_config_prop_group, 'IncludePath')
        ET.SubElement(per_config_prop_group, 'ExternalIncludePath')
        ET.SubElement(per_config_prop_group, 'ReferencePath')
        ET.SubElement(per_config_prop_group, 'LibraryPath')
        ET.SubElement(per_config_prop_group, 'LibraryWPath')
        ET.SubElement(per_config_prop_group, 'SourcePath')
        ET.SubElement(per_config_prop_group, 'ExcludePath')