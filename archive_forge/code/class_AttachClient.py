import errno
import fcntl
import os
from oslo_log import log as logging
import select
import signal
import socket
import ssl
import struct
import sys
import termios
import time
import tty
from urllib import parse as urlparse
import websocket
from zunclient.common.apiclient import exceptions as acexceptions
from zunclient.common.websocketclient import exceptions
class AttachClient(WebSocketClient):

    def tty_resize(self, height, width):
        """Resize the tty session

        Get the client and send the tty size data to zun api server
        The environment variables need to get when implement sending
        operation.
        """
        height = str(height)
        width = str(width)
        self.cs.containers.resize(self.id, width, height)