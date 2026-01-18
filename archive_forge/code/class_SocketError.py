import errno
import os
import select
import socket as pysocket
import struct
class SocketError(Exception):
    pass