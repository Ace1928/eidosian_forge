from neutron_lib.api.definitions import logging as log_api
from neutron_lib.tests.unit.api.definitions import base
class LoggingApiTestCase(base.DefinitionBaseTestCase):
    extension_module = log_api
    extension_attributes = EXTENSION_ATTRIBUTES
    extension_resources = (log_api.LOGS, log_api.LOG_TYPES)