import sys
from os_win._i18n import _
class PciDeviceNotFound(NotFound):
    msg_fmt = _('No assignable PCI device with vendor id: %(vendor_id)s and product id: %(product_id)s was found.')