from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def _MakeDispatchListIntoYaml(application, dispatch_list):
    """Converts list of DispatchEntry objects into a YAML string."""
    statements = []
    if application:
        statements.append('application: %s' % application)
    statements.append('dispatch:')
    for entry in dispatch_list:
        statements += entry.ToYaml()
    return '\n'.join(statements) + '\n'