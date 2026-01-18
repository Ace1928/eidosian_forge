from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class PushQueue(Queue):

    def GetAdditionalYamlStatementsList(self):
        statements = ['  mode: push']
        fields = (tag.replace('-', '_') for tag in PUSH_QUEUE_TAGS)
        for field in fields:
            field_value = getattr(self, field)
            if field_value:
                statements.append('  %s: %s' % (field, field_value))
        return statements