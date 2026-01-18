from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
def GetYamlStatementsList(self):
    """Converts retry parameter fields to a YAML statement list."""
    tag_statements = []
    field_names = (tag.replace('-', '_') for tag in _RETRY_PARAMETER_TAGS)
    for field in field_names:
        field_value = getattr(self, field, None)
        if field_value:
            tag_statements.append('    %s: %s' % (field, field_value))
    if not tag_statements:
        return ['  retry_parameters: {}']
    return ['  retry_parameters:'] + tag_statements