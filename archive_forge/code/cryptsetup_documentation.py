import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_log import versionutils
from os_brick.encryptors import base
from os_brick import exception
Removes the dm-crypt mapping for the device.