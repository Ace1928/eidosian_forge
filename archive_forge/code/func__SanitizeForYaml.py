from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
def _SanitizeForYaml(self, field):
    return "'%s'" % field.replace('\n', ' ').replace("'", "''")