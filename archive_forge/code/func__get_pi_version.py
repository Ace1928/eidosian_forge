from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
def _get_pi_version():
    """Detect the version of the Raspberry Pi by reading the revision field value from '/proc/cpuinfo'
    See: https://www.raspberrypi.org/documentation/hardware/raspberrypi/revision-codes/README.md
    Based on: https://github.com/adafruit/Adafruit_Python_GPIO/blob/master/Adafruit_GPIO/Platform.py
    """
    if not path.isfile('/proc/cpuinfo'):
        return None
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    revision = search('^Revision\\s+:\\s+(\\w+)$', cpuinfo, flags=MULTILINE | IGNORECASE)
    if not revision:
        return None
    revision = int(revision.group(1), base=16)
    if revision & 8388608:
        return ((revision & 61440) >> 12) + 1
    return 1