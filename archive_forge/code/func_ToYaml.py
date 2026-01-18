from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
def ToYaml(self):
    """Returns data from Cron object as a list of Yaml statements."""
    statements = ['- url: %s' % self._SanitizeForYaml(self.url), '  schedule: %s' % self._SanitizeForYaml(self.schedule)]
    for optional in ('target', 'timezone', 'description'):
        field = getattr(self, optional)
        if field:
            statements.append('  %s: %s' % (optional, self._SanitizeForYaml(field)))
    retry_parameters = getattr(self, 'retry_parameters', None)
    if retry_parameters:
        statements += retry_parameters.GetYamlStatementsList()
    return statements