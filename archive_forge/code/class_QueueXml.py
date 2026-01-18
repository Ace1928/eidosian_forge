from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class QueueXml(object):

    def __init__(self):
        self.queues = []
        self.total_storage_limit = None

    def ToYaml(self):
        statements = []
        if self.total_storage_limit:
            statements.append('total_storage_limit: %s\n' % self.total_storage_limit)
        statements.append('queue:')
        for queue in self.queues:
            statements += queue.GetYamlStatementsList()
        return '\n'.join(statements) + '\n'