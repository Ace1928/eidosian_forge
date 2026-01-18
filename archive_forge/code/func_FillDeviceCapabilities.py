import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
def FillDeviceCapabilities(device, descriptor):
    """Fill out device capabilities.

  Fills the HidCapabilitites of the device into descriptor.

  Args:
    device: A handle to the open device
    descriptor: DeviceDescriptor to populate with the
      capabilities

  Returns:
    none

  Raises:
    WindowsError when unable to obtain capabilitites.
  """
    preparsed_data = PHIDP_PREPARSED_DATA(0)
    ret = hid.HidD_GetPreparsedData(device, ctypes.byref(preparsed_data))
    if not ret:
        raise ctypes.WinError()
    try:
        caps = HidCapabilities()
        ret = hid.HidP_GetCaps(preparsed_data, ctypes.byref(caps))
        if ret != HIDP_STATUS_SUCCESS:
            raise ctypes.WinError()
        descriptor.usage = caps.Usage
        descriptor.usage_page = caps.UsagePage
        descriptor.internal_max_in_report_len = caps.InputReportByteLength
        descriptor.internal_max_out_report_len = caps.OutputReportByteLength
    finally:
        hid.HidD_FreePreparsedData(preparsed_data)