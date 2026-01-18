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
def _AddMSBuildAction(spec, primary_input, inputs, outputs, cmd, description, sources_handled_by_action, actions_spec):
    command = MSVSSettings.ConvertVCMacrosToMSBuild(cmd)
    primary_input = _FixPath(primary_input)
    inputs_array = _FixPaths(inputs)
    outputs_array = _FixPaths(outputs)
    additional_inputs = ';'.join([i for i in inputs_array if i != primary_input])
    outputs = ';'.join(outputs_array)
    sources_handled_by_action.add(primary_input)
    action_spec = ['CustomBuild', {'Include': primary_input}]
    action_spec.extend([['FileType', 'Document'], ['Command', command], ['Message', description], ['Outputs', outputs]])
    if additional_inputs:
        action_spec.append(['AdditionalInputs', additional_inputs])
    actions_spec.append(action_spec)