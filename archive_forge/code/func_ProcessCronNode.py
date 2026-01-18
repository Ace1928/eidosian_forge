from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
def ProcessCronNode(self, node):
    """Processes XML <cron> nodes into Cron objects.

    The following information is parsed out:
      description: Describing the purpose of the cron job.
      url: The location of the script.
      schedule: Written in groc; the schedule according to which the job is
        executed.
      timezone: The timezone that the schedule runs in.
      target: Which version of the app this applies to.

    Args:
      node: <cron> XML node in cron.xml.
    """
    tag = xml_parser_utils.GetTag(node)
    if tag != 'cron':
        self.errors.append('Unrecognized node: <%s>' % tag)
        return
    cron = Cron()
    cron.url = xml_parser_utils.GetChildNodeText(node, 'url')
    cron.timezone = xml_parser_utils.GetChildNodeText(node, 'timezone')
    cron.target = xml_parser_utils.GetChildNodeText(node, 'target')
    cron.description = xml_parser_utils.GetChildNodeText(node, 'description')
    cron.schedule = xml_parser_utils.GetChildNodeText(node, 'schedule')
    _ProcessRetryParametersNode(node, cron)
    validation_error = self._ValidateCronEntry(cron)
    if validation_error:
        self.errors.append(validation_error)
    else:
        self.crons.append(cron)