import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def _get_assignable_pci_device(self, vendor_id, product_id):
    pci_devices = self._conn.Msvm_PciExpress()
    pattern = re.compile('^(.*)VEN_%(vendor_id)s&DEV_%(product_id)s&(.*)$' % {'vendor_id': vendor_id, 'product_id': product_id}, re.IGNORECASE)
    for dev in pci_devices:
        if pattern.match(dev.DeviceID):
            pci_devices_found = [d for d in pci_devices if d.LocationPath == dev.LocationPath]
            LOG.debug('PCI devices found: %s', [d.DeviceID for d in pci_devices_found])
            if len(pci_devices_found) == 1:
                return pci_devices_found[0]
    raise exceptions.PciDeviceNotFound(vendor_id=vendor_id, product_id=product_id)