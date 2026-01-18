from __future__ import absolute_import
import glob
import os
from serial.tools import list_ports_common
class SysFS(list_ports_common.ListPortInfo):
    """Wrapper for easy sysfs access and device info"""

    def __init__(self, device):
        super(SysFS, self).__init__(device)
        if device is not None and os.path.islink(device):
            device = os.path.realpath(device)
            is_link = True
        else:
            is_link = False
        self.usb_device_path = None
        if os.path.exists('/sys/class/tty/{}/device'.format(self.name)):
            self.device_path = os.path.realpath('/sys/class/tty/{}/device'.format(self.name))
            self.subsystem = os.path.basename(os.path.realpath(os.path.join(self.device_path, 'subsystem')))
        else:
            self.device_path = None
            self.subsystem = None
        if self.subsystem == 'usb-serial':
            self.usb_interface_path = os.path.dirname(self.device_path)
        elif self.subsystem == 'usb':
            self.usb_interface_path = self.device_path
        else:
            self.usb_interface_path = None
        if self.usb_interface_path is not None:
            self.usb_device_path = os.path.dirname(self.usb_interface_path)
            try:
                num_if = int(self.read_line(self.usb_device_path, 'bNumInterfaces'))
            except ValueError:
                num_if = 1
            self.vid = int(self.read_line(self.usb_device_path, 'idVendor'), 16)
            self.pid = int(self.read_line(self.usb_device_path, 'idProduct'), 16)
            self.serial_number = self.read_line(self.usb_device_path, 'serial')
            if num_if > 1:
                self.location = os.path.basename(self.usb_interface_path)
            else:
                self.location = os.path.basename(self.usb_device_path)
            self.manufacturer = self.read_line(self.usb_device_path, 'manufacturer')
            self.product = self.read_line(self.usb_device_path, 'product')
            self.interface = self.read_line(self.usb_interface_path, 'interface')
        if self.subsystem in ('usb', 'usb-serial'):
            self.apply_usb_info()
        elif self.subsystem == 'pnp':
            self.description = self.name
            self.hwid = self.read_line(self.device_path, 'id')
        elif self.subsystem == 'amba':
            self.description = self.name
            self.hwid = os.path.basename(self.device_path)
        if is_link:
            self.hwid += ' LINK={}'.format(device)

    def read_line(self, *args):
        """        Helper function to read a single line from a file.
        One or more parameters are allowed, they are joined with os.path.join.
        Returns None on errors..
        """
        try:
            with open(os.path.join(*args)) as f:
                line = f.readline().strip()
            return line
        except IOError:
            return None