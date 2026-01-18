from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def _parse_coroutine(self):
    """
        Parser state machine.
        Every 'yield' expression returns the next byte.
        """
    while True:
        d = (yield)
        if d == int2byte(0):
            pass
        elif d == IAC:
            d2 = (yield)
            if d2 == IAC:
                self.received_data(d2)
            elif d2 in (NOP, DM, BRK, IP, AO, AYT, EC, EL, GA):
                self.command_received(d2, None)
            elif d2 in (DO, DONT, WILL, WONT):
                d3 = (yield)
                self.command_received(d2, d3)
            elif d2 == SB:
                data = []
                while True:
                    d3 = (yield)
                    if d3 == IAC:
                        d4 = (yield)
                        if d4 == SE:
                            break
                        else:
                            data.append(d4)
                    else:
                        data.append(d3)
                self.negotiate(b''.join(data))
        else:
            self.received_data(d)