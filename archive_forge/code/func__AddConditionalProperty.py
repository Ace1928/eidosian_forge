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
def _AddConditionalProperty(properties, condition, name, value):
    """Adds a property / conditional value pair to a dictionary.

  Arguments:
    properties: The dictionary to be modified.  The key is the name of the
        property.  The value is itself a dictionary; its key is the value and
        the value a list of condition for which this value is true.
    condition: The condition under which the named property has the value.
    name: The name of the property.
    value: The value of the property.
  """
    if name not in properties:
        properties[name] = {}
    values = properties[name]
    if value not in values:
        values[value] = []
    conditions = values[value]
    conditions.append(condition)