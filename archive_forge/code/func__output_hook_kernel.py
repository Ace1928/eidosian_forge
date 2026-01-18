import asyncio
import inspect
import sys
import time
import typing as t
from functools import partial
from getpass import getpass
from queue import Empty
import zmq.asyncio
from jupyter_core.utils import ensure_async
from traitlets import Any, Bool, Instance, Type
from .channels import major_protocol_version
from .channelsabc import ChannelABC, HBChannelABC
from .clientabc import KernelClientABC
from .connect import ConnectionFileMixin
from .session import Session
def _output_hook_kernel(self, session: Session, socket: zmq.sugar.socket.Socket, parent_header: t.Any, msg: t.Dict[str, t.Any]) -> None:
    """Output hook when running inside an IPython kernel

        adds rich output support.
        """
    msg_type = msg['header']['msg_type']
    if msg_type in ('display_data', 'execute_result', 'error'):
        session.send(socket, msg_type, msg['content'], parent=parent_header)
    else:
        self._output_hook_default(msg)