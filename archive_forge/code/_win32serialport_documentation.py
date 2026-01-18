import win32event
import win32file
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE
from serial.serialutil import to_bytes
from twisted.internet import abstract
from twisted.internet.serialport import BaseSerialPort

        Called when the serial port disconnects.

        Will call C{connectionLost} on the protocol that is handling the
        serial data.
        