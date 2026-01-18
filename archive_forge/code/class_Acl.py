from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class Acl(object):

    def GetYamlStatementsList(self):
        statements = ['  acl:']
        statements += ['  - user_email: %s' % user_email for user_email in self.user_emails]
        statements += ['  - writer_email: %s' % writer_email for writer_email in self.writer_emails]
        return statements