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
def _AppendFiltersForMSBuild(parent_filter_name, sources, rule_dependencies, extension_to_rule_name, platforms, toolset, filter_group, source_group):
    """Creates the list of filters and sources to be added in the filter file.

  Args:
      parent_filter_name: The name of the filter under which the sources are
          found.
      sources: The hierarchy of filters and sources to process.
      extension_to_rule_name: A dictionary mapping file extensions to rules.
      filter_group: The list to which filter entries will be appended.
      source_group: The list to which source entries will be appended.
  """
    for source in sources:
        if isinstance(source, MSVSProject.Filter):
            if not parent_filter_name:
                filter_name = source.name
            else:
                filter_name = f'{parent_filter_name}\\{source.name}'
            filter_group.append(['Filter', {'Include': filter_name}, ['UniqueIdentifier', MSVSNew.MakeGuid(source.name)]])
            _AppendFiltersForMSBuild(filter_name, source.contents, rule_dependencies, extension_to_rule_name, platforms, toolset, filter_group, source_group)
        else:
            _, element = _MapFileToMsBuildSourceType(source, rule_dependencies, extension_to_rule_name, platforms, toolset)
            source_entry = [element, {'Include': source}]
            if parent_filter_name:
                source_entry.append(['Filter', parent_filter_name])
            source_group.append(source_entry)