from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
@staticmethod
def _get_led_locate_uri(data):
    """Get the LED locate URI given a resource body."""
    if WdcRedfishUtils.ACTIONS not in data:
        return None
    if WdcRedfishUtils.OEM not in data[WdcRedfishUtils.ACTIONS]:
        return None
    if WdcRedfishUtils.WDC not in data[WdcRedfishUtils.ACTIONS][WdcRedfishUtils.OEM]:
        return None
    if WdcRedfishUtils.CHASSIS_LOCATE not in data[WdcRedfishUtils.ACTIONS][WdcRedfishUtils.OEM][WdcRedfishUtils.WDC]:
        return None
    if WdcRedfishUtils.TARGET not in data[WdcRedfishUtils.ACTIONS][WdcRedfishUtils.OEM][WdcRedfishUtils.WDC][WdcRedfishUtils.CHASSIS_LOCATE]:
        return None
    return data[WdcRedfishUtils.ACTIONS][WdcRedfishUtils.OEM][WdcRedfishUtils.WDC][WdcRedfishUtils.CHASSIS_LOCATE][WdcRedfishUtils.TARGET]