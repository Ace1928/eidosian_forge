from __future__ import absolute_import
import time
import serial
class RS485Settings(object):

    def __init__(self, rts_level_for_tx=True, rts_level_for_rx=False, loopback=False, delay_before_tx=None, delay_before_rx=None):
        self.rts_level_for_tx = rts_level_for_tx
        self.rts_level_for_rx = rts_level_for_rx
        self.loopback = loopback
        self.delay_before_tx = delay_before_tx
        self.delay_before_rx = delay_before_rx