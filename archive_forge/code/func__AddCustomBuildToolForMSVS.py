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
def _AddCustomBuildToolForMSVS(p, spec, primary_input, inputs, outputs, description, cmd):
    """Add a custom build tool to execute something.

  Arguments:
    p: the target project
    spec: the target project dict
    primary_input: input file to attach the build tool to
    inputs: list of inputs
    outputs: list of outputs
    description: description of the action
    cmd: command line to execute
  """
    inputs = _FixPaths(inputs)
    outputs = _FixPaths(outputs)
    tool = MSVSProject.Tool('VCCustomBuildTool', {'Description': description, 'AdditionalDependencies': ';'.join(inputs), 'Outputs': ';'.join(outputs), 'CommandLine': cmd})
    for config_name, c_data in spec['configurations'].items():
        p.AddFileConfig(_FixPath(primary_input), _ConfigFullName(config_name, c_data), tools=[tool])