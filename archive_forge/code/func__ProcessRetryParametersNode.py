from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
def _ProcessRetryParametersNode(node, cron):
    """Converts <retry-parameters> in node to cron.retry_parameters."""
    retry_parameters_node = xml_parser_utils.GetChild(node, 'retry-parameters')
    if retry_parameters_node is None:
        cron.retry_parameters = None
        return
    retry_parameters = _RetryParameters()
    cron.retry_parameters = retry_parameters
    for tag in _RETRY_PARAMETER_TAGS:
        if xml_parser_utils.GetChild(retry_parameters_node, tag) is not None:
            setattr(retry_parameters, tag.replace('-', '_'), xml_parser_utils.GetChildNodeText(retry_parameters_node, tag))