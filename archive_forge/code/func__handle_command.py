from __future__ import unicode_literals
import socket
import select
import threading
import os
import fcntl
from six import int2byte, text_type, binary_type
from codecs import getincrementaldecoder
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.eventloop.base import EventLoop
from prompt_toolkit.interface import CommandLineInterface, Application
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.shortcuts import create_prompt_application
from prompt_toolkit.terminal.vt100_input import InputStream
from prompt_toolkit.terminal.vt100_output import Vt100_Output
from .log import logger
from .protocol import IAC, DO, LINEMODE, SB, MODE, SE, WILL, ECHO, NAWS, SUPPRESS_GO_AHEAD
from .protocol import TelnetProtocolParser
from .application import TelnetApplication
def _handle_command(self, command):
    """
        Handle command. This will run in a separate thread, in order not
        to block the event loop.
        """
    logger.info('Handle command %r', command)

    def in_executor():
        self.handling_command = True
        try:
            if self.callback is not None:
                self.callback(self, command)
        finally:
            self.server.call_from_executor(done)

    def done():
        self.handling_command = False
        if not self.closed:
            self.cli.reset()
            self.cli.buffers[DEFAULT_BUFFER].reset()
            self.cli.renderer.request_absolute_cursor_position()
            self.vt100_output.flush()
            self.cli._redraw()
    self.server.run_in_executor(in_executor)