from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
class CronXmlParser(object):
    """Provides logic for walking down XML tree and pulling data."""

    def ProcessXml(self, xml_str):
        """Parses XML string and returns object representation of relevant info.

    Args:
      xml_str: The XML string.
    Returns:
      A list of Cron objects containing information about cron jobs from the
      XML.
    Raises:
      AppEngineConfigException: In case of malformed XML or illegal inputs.
    """
        try:
            self.crons = []
            self.errors = []
            xml_root = ElementTree.fromstring(xml_str)
            if xml_root.tag != 'cronentries':
                raise AppEngineConfigException('Root tag must be <cronentries>')
            for child in list(xml_root):
                self.ProcessCronNode(child)
            if self.errors:
                raise AppEngineConfigException('\n'.join(self.errors))
            return self.crons
        except ElementTree.ParseError:
            raise AppEngineConfigException('Bad input -- not valid XML')

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

    def _ValidateCronEntry(self, cron):
        if not cron.url:
            return 'No URL for <cron> entry'
        if not cron.schedule:
            return "No schedule provided for <cron> entry with URL '%s'" % cron.url
        if groc and groctimespecification:
            try:
                groctimespecification.GrocTimeSpecification(cron.schedule)
            except groc.GrocException:
                return "Text '%s' in <schedule> node failed to parse, for <cron> entry with url %s." % (cron.schedule, cron.url)