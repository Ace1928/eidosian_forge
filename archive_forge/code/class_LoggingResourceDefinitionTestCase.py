from neutron_lib.api.definitions import logging_resource
from neutron_lib.tests.unit.api.definitions import base
class LoggingResourceDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = logging_resource
    extension_resources = (logging_resource.COLLECTION_NAME,)
    extension_subresources = (logging_resource.FIREWALL_LOGS,)
    extension_attributes = (logging_resource.ENABLED, logging_resource.FIREWALL_LOGS, logging_resource.LOGGING_RESOURCE_ID, logging_resource.FW_EVENT, logging_resource.FIREWALL_ID)