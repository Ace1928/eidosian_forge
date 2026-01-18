from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class DispatchEntry(object):
    """Instances contain information about individual dispatch entries."""

    def ToYaml(self):
        return ["- url: '%s'" % self._SanitizeForYaml(self.url), '  module: %s' % self.module]

    def _SanitizeForYaml(self, dirty_str):
        return dirty_str.replace("'", "\\'")