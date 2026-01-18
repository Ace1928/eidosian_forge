from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def LoadAppInclude(app_include):
    """Loads a single `AppInclude` object where one and only one is expected.

  Args:
    app_include: A file-like object or string. The argument is set to a string,
        the argument is parsed as a configuration file. If the argument is set
        to a file-like object, the data is read and parsed.

  Returns:
    An instance of `AppInclude` as loaded from a YAML file.

  Raises:
    EmptyConfigurationFile: If there are no documents in the YAML file.
    MultipleConfigurationFile: If there is more than one document in the YAML
        file.
  """
    builder = yaml_object.ObjectBuilder(AppInclude)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(app_include)
    includes = handler.GetResults()
    if len(includes) < 1:
        raise appinfo_errors.EmptyConfigurationFile()
    if len(includes) > 1:
        raise appinfo_errors.MultipleConfigurationFile()
    includeyaml = includes[0]
    if includeyaml.handlers:
        for handler in includeyaml.handlers:
            handler.FixSecureDefaults()
            handler.WarnReservedURLs()
    if includeyaml.builtins:
        BuiltinHandler.Validate(includeyaml.builtins)
    return includeyaml