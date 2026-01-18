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
def LoadSingleAppInfo(app_info):
    """Loads a single `AppInfo` object where one and only one is expected.

  This method validates that the values in the `AppInfo` match the
  validators that are defined in this file, in particular,
  `AppInfoExternal.ATTRIBUTES`.

  Args:
    app_info: A file-like object or string. If the argument is a string, the
        argument is parsed as a configuration file. If the argument is a
        file-like object, the data is read, then parsed.

  Returns:
    An instance of `AppInfoExternal` as loaded from a YAML file.

  Raises:
    ValueError: If a specified service is not valid.
    EmptyConfigurationFile: If there are no documents in YAML file.
    MultipleConfigurationFile: If more than one document exists in the YAML
        file.
    DuplicateBackend: If a backend is found more than once in the `backends`
        directive.
    yaml_errors.EventError: If the `app.yaml` file fails validation.
    appinfo_errors.MultipleProjectNames: If the `app.yaml` file has both an
        `application` directive and a `project` directive.
  """
    builder = yaml_object.ObjectBuilder(AppInfoExternal)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(app_info)
    app_infos = handler.GetResults()
    if len(app_infos) < 1:
        raise appinfo_errors.EmptyConfigurationFile()
    if len(app_infos) > 1:
        raise appinfo_errors.MultipleConfigurationFile()
    appyaml = app_infos[0]
    ValidateHandlers(appyaml.handlers)
    if appyaml.builtins:
        BuiltinHandler.Validate(appyaml.builtins, appyaml.runtime)
    if appyaml.application and appyaml.project:
        raise appinfo_errors.MultipleProjectNames('Specify one of "application: name" or "project: name"')
    elif appyaml.project:
        appyaml.application = appyaml.project
        appyaml.project = None
    appyaml.NormalizeVmSettings()
    return appyaml