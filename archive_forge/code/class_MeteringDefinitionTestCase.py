from neutron_lib.api.definitions import metering
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests.unit.api.definitions import base
class MeteringDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = metering
    extension_resources = (metering.METERING_LABEL_RULES, metering.METERING_LABELS)
    extension_attributes = ('remote_ip_prefix', 'excluded', 'metering_label_id', qos_consts.DIRECTION)