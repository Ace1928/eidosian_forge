from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
Processes XML <dispatch> nodes into DispatchEntry objects.

    The following information is parsed out:
      url: The URL or URL pattern to route.
      module: The module to route it to.
    If there are no errors, the data is loaded into a DispatchEntry object
    and added to a list. Upon error, a description of the error is added to
    a list and the method terminates.

    Args:
      node: <dispatch> XML node in dos.xml.
    