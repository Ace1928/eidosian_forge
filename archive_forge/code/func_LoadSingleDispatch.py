from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
def LoadSingleDispatch(dispatch_info, open_fn=None):
    """Load a dispatch.yaml file or string and return a DispatchInfoExternal.

  Args:
    dispatch_info: The contents of a dispatch.yaml file as a string, or an open
      file object.
    open_fn: Function for opening files. Unused here, needed to provide
      a polymorphic API used by appcfg.py yaml parsing.

  Returns:
    A DispatchInfoExternal instance which represents the contents of the parsed
      yaml file.

  Raises:
    MalformedDispatchConfigurationError: The yaml file contains multiple
      dispatch sections or is missing a required value.
    yaml_errors.EventError: An error occured while parsing the yaml file.
  """
    builder = yaml_object.ObjectBuilder(DispatchInfoExternal)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(dispatch_info)
    parsed_yaml = handler.GetResults()
    if not parsed_yaml:
        return DispatchInfoExternal()
    if len(parsed_yaml) > 1:
        raise MalformedDispatchConfigurationError('Multiple dispatch: sections in configuration.')
    dispatch_info_external = parsed_yaml[0]
    dispatch_info_external.CheckInitialized()
    return dispatch_info_external