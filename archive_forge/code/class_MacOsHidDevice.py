from six.moves import queue
from six.moves import range
import ctypes
import ctypes.util
import logging
import sys
import threading
from pyu2f import errors
from pyu2f.hid import base
class MacOsHidDevice(base.HidDevice):
    """Implementation of HID device for MacOS.

  Uses IOKit HID Manager to interact with the device.
  """

    @staticmethod
    def Enumerate():
        """See base class."""
        hid_mgr = iokit.IOHIDManagerCreate(None, None)
        if not hid_mgr:
            raise errors.OsHidError('Unable to obtain HID manager reference')
        iokit.IOHIDManagerSetDeviceMatching(hid_mgr, None)
        device_set_ref = iokit.IOHIDManagerCopyDevices(hid_mgr)
        if not device_set_ref:
            raise errors.OsHidError('Failed to obtain devices from HID manager')
        num = iokit.CFSetGetCount(device_set_ref)
        devices = (IO_HID_DEVICE_REF * num)()
        iokit.CFSetGetValues(device_set_ref, devices)
        descriptors = []
        for dev in devices:
            d = base.DeviceDescriptor()
            d.vendor_id = GetDeviceIntProperty(dev, HID_DEVICE_PROPERTY_VENDOR_ID)
            d.product_id = GetDeviceIntProperty(dev, HID_DEVICE_PROPERTY_PRODUCT_ID)
            d.product_string = GetDeviceStringProperty(dev, HID_DEVICE_PROPERTY_PRODUCT)
            d.usage = GetDeviceIntProperty(dev, HID_DEVICE_PROPERTY_PRIMARY_USAGE)
            d.usage_page = GetDeviceIntProperty(dev, HID_DEVICE_PROPERTY_PRIMARY_USAGE_PAGE)
            d.report_id = GetDeviceIntProperty(dev, HID_DEVICE_PROPERTY_REPORT_ID)
            d.path = GetDevicePath(dev)
            descriptors.append(d.ToPublicDict())
        cf.CFRelease(device_set_ref)
        cf.CFRelease(hid_mgr)
        return descriptors

    def __init__(self, path):
        device_entry = iokit.IORegistryEntryFromPath(K_IO_MASTER_PORT_DEFAULT, path)
        if not device_entry:
            raise errors.OsHidError('Device path does not match any HID device on the system')
        self.device_handle = iokit.IOHIDDeviceCreate(K_CF_ALLOCATOR_DEFAULT, device_entry)
        if not self.device_handle:
            raise errors.OsHidError('Failed to obtain device handle from registry entry')
        iokit.IOObjectRelease(device_entry)
        self.device_path = path
        result = iokit.IOHIDDeviceOpen(self.device_handle, 0)
        if result != K_IO_RETURN_SUCCESS:
            raise errors.OsHidError('Failed to open device for communication: {}'.format(result))
        self.read_queue = queue.Queue()
        self.run_loop_ref = None
        self.read_thread = threading.Thread(target=DeviceReadThread, args=(self,))
        self.read_thread.daemon = True
        self.read_thread.start()
        self.internal_max_in_report_len = GetDeviceIntProperty(self.device_handle, HID_DEVICE_PROPERTY_MAX_INPUT_REPORT_SIZE)
        if not self.internal_max_in_report_len:
            raise errors.OsHidError('Unable to obtain max in report size')
        self.internal_max_out_report_len = GetDeviceIntProperty(self.device_handle, HID_DEVICE_PROPERTY_MAX_OUTPUT_REPORT_SIZE)
        if not self.internal_max_out_report_len:
            raise errors.OsHidError('Unable to obtain max out report size')
        self.in_report_buffer = (ctypes.c_uint8 * self.internal_max_in_report_len)()
        iokit.IOHIDDeviceRegisterInputReportCallback(self.device_handle, self.in_report_buffer, self.internal_max_in_report_len, REGISTERED_READ_CALLBACK, ctypes.py_object(self.read_queue))

    def GetInReportDataLength(self):
        """See base class."""
        return self.internal_max_in_report_len

    def GetOutReportDataLength(self):
        """See base class."""
        return self.internal_max_out_report_len

    def Write(self, packet):
        """See base class."""
        report_id = 0
        out_report_buffer = (ctypes.c_uint8 * self.internal_max_out_report_len)()
        out_report_buffer[:] = packet[:]
        result = iokit.IOHIDDeviceSetReport(self.device_handle, K_IO_HID_REPORT_TYPE_OUTPUT, report_id, out_report_buffer, self.internal_max_out_report_len)
        if result != K_IO_RETURN_SUCCESS:
            raise errors.OsHidError('Failed to write report to device')

    def Read(self):
        """See base class."""
        result = None
        while result is None:
            try:
                result = self.read_queue.get(timeout=60)
            except queue.Empty:
                continue
        return result

    def __del__(self):
        if hasattr(self, 'in_report_buffer'):
            iokit.IOHIDDeviceRegisterInputReportCallback(self.device_handle, self.in_report_buffer, self.internal_max_in_report_len, None, None)
        if hasattr(self, 'run_loop_ref'):
            cf.CFRunLoopStop(self.run_loop_ref)
        if hasattr(self, 'read_thread'):
            self.read_thread.join()