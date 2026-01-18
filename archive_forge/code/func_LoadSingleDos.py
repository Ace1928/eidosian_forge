from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
def LoadSingleDos(dos_info, open_fn=None):
    """Load a dos.yaml file or string and return a DosInfoExternal object.

  Args:
    dos_info: The contents of a dos.yaml file as a string, or an open file
      object.
    open_fn: Function for opening files. Unused.

  Returns:
    A DosInfoExternal instance which represents the contents of the parsed yaml
    file.

  Raises:
    MalformedDosConfiguration: The yaml file contains multiple blacklist
      sections.
    yaml_errors.EventError: An error occured while parsing the yaml file.
  """
    builder = yaml_object.ObjectBuilder(DosInfoExternal)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(dos_info)
    parsed_yaml = handler.GetResults()
    if not parsed_yaml:
        return DosInfoExternal()
    if len(parsed_yaml) > 1:
        raise MalformedDosConfiguration('Multiple blacklist: sections in configuration.')
    return parsed_yaml[0]