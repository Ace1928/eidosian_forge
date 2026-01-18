from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
class TelnetSubnegotiation(object):
    """    A object to handle subnegotiation of options. In this case actually
    sub-sub options for RFC 2217. It is used to track com port options.
    """

    def __init__(self, connection, name, option, ack_option=None):
        if ack_option is None:
            ack_option = option
        self.connection = connection
        self.name = name
        self.option = option
        self.value = None
        self.ack_option = ack_option
        self.state = INACTIVE

    def __repr__(self):
        """String for debug outputs."""
        return '{sn.name}:{sn.state}'.format(sn=self)

    def set(self, value):
        """        Request a change of the value. a request is sent to the server. if
        the client needs to know if the change is performed he has to check the
        state of this object.
        """
        self.value = value
        self.state = REQUESTED
        self.connection.rfc2217_send_subnegotiation(self.option, self.value)
        if self.connection.logger:
            self.connection.logger.debug('SB Requesting {} -> {!r}'.format(self.name, self.value))

    def is_ready(self):
        """        Check if answer from server has been received. when server rejects
        the change, raise a ValueError.
        """
        if self.state == REALLY_INACTIVE:
            raise ValueError('remote rejected value for option {!r}'.format(self.name))
        return self.state == ACTIVE
    active = property(is_ready)

    def wait(self, timeout=3):
        """        Wait until the subnegotiation has been acknowledged or timeout. It
        can also throw a value error when the answer from the server does not
        match the value sent.
        """
        timeout_timer = Timeout(timeout)
        while not timeout_timer.expired():
            time.sleep(0.05)
            if self.is_ready():
                break
        else:
            raise SerialException('timeout while waiting for option {!r}'.format(self.name))

    def check_answer(self, suboption):
        """        Check an incoming subnegotiation block. The parameter already has
        cut off the header like sub option number and com port option value.
        """
        if self.value == suboption[:len(self.value)]:
            self.state = ACTIVE
        else:
            self.state = REALLY_INACTIVE
        if self.connection.logger:
            self.connection.logger.debug('SB Answer {} -> {!r} -> {}'.format(self.name, suboption, self.state))