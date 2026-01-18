import os
import subprocess
import sys
from debugpy import adapter, common
from debugpy.common import log, messaging, sockets
from debugpy.adapter import components, servers, sessions
class Launcher(components.Component):
    """Handles the launcher side of a debug session."""
    message_handler = components.Component.message_handler

    def __init__(self, session, stream):
        with session:
            assert not session.launcher
            super().__init__(session, stream)
            self.pid = None
            'Process ID of the debuggee process, as reported by the launcher.'
            self.exit_code = None
            'Exit code of the debuggee process.'
            session.launcher = self

    @message_handler
    def process_event(self, event):
        self.pid = event('systemProcessId', int)
        self.client.propagate_after_start(event)

    @message_handler
    def output_event(self, event):
        self.client.propagate_after_start(event)

    @message_handler
    def exited_event(self, event):
        self.exit_code = event('exitCode', int)

    @message_handler
    def terminated_event(self, event):
        try:
            self.client.channel.send_event('exited', {'exitCode': self.exit_code})
        except Exception:
            pass
        self.channel.close()

    def terminate_debuggee(self):
        with self.session:
            if self.exit_code is None:
                try:
                    self.channel.request('terminate')
                except Exception:
                    pass