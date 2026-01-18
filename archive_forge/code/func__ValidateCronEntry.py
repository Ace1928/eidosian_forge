from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
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