import platform
import sys
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_service import loopingcall
from os_brick import exception
from os_brick.initiator import connector
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fake
from os_brick.initiator.connectors import iscsi
from os_brick.initiator.connectors import nvmeof
from os_brick.initiator import linuxfc
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick import utils
class ZeroIntervalLoopingCall(loopingcall.FixedIntervalLoopingCall):

    def start(self, interval, initial_delay=None, stop_on_exception=True):
        return super(ZeroIntervalLoopingCall, self).start(0, 0, stop_on_exception)