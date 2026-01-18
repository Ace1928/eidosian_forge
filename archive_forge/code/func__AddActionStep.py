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
def _AddActionStep(actions_dict, inputs, outputs, description, command):
    """Merge action into an existing list of actions.

  Care must be taken so that actions which have overlapping inputs either don't
  get assigned to the same input, or get collapsed into one.

  Arguments:
    actions_dict: dictionary keyed on input name, which maps to a list of
      dicts describing the actions attached to that input file.
    inputs: list of inputs
    outputs: list of outputs
    description: description of the action
    command: command line to execute
  """
    assert inputs
    action = {'inputs': inputs, 'outputs': outputs, 'description': description, 'command': command}
    chosen_input = inputs[0]
    if chosen_input not in actions_dict:
        actions_dict[chosen_input] = []
    actions_dict[chosen_input].append(action)