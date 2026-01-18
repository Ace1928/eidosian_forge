from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
def _convert_microseconds(micros):
    seconds = micros / 1000 % 60
    minutes = micros / (1000 * 60) % 60
    hours = micros / (1000 * 60 * 60) % 24
    return (seconds, minutes, hours)