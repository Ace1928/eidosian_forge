from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.ext import builtins
def ParseAndReturnIncludePaths(appinfo_file):
    """Parse an AppYaml file and merge referenced includes and builtins.

  Args:
    appinfo_file: an opened file, for example the result of open('app.yaml').

  Returns:
    A tuple where the first element is the parsed appinfo.AppInfoExternal
    object and the second element is a list of the absolute paths of the
    included files, in no particular order.
  """
    try:
        appinfo_path = appinfo_file.name
        if not os.path.isfile(appinfo_path):
            raise Exception('Name defined by appinfo_file does not appear to be a valid file: %s' % appinfo_path)
    except AttributeError:
        raise Exception('File object passed to ParseAndMerge does not define attribute "name" as as full file path.')
    appyaml = appinfo.LoadSingleAppInfo(appinfo_file)
    appyaml, include_paths = _MergeBuiltinsIncludes(appinfo_path, appyaml)
    if not appyaml.handlers:
        if appyaml.IsVm():
            appyaml.handlers = [appinfo.URLMap(url='.*', script='PLACEHOLDER')]
        else:
            appyaml.handlers = []
    if len(appyaml.handlers) > appinfo.MAX_URL_MAPS:
        raise appinfo_errors.TooManyURLMappings('Found more than %d URLMap entries in application configuration' % appinfo.MAX_URL_MAPS)
    if appyaml.runtime == 'python27' and appyaml.threadsafe:
        for handler in appyaml.handlers:
            if handler.script and (handler.script.endswith('.py') or '/' in handler.script):
                raise appinfo_errors.ThreadsafeWithCgiHandler('Threadsafe cannot be enabled with CGI handler: %s' % handler.script)
    return (appyaml, include_paths)