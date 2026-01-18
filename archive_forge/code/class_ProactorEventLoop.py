import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
class ProactorEventLoop(proactor_events.BaseProactorEventLoop):
    """Windows version of proactor event loop using IOCP."""

    def __init__(self, proactor=None):
        if proactor is None:
            proactor = IocpProactor()
        super().__init__(proactor)

    def run_forever(self):
        try:
            assert self._self_reading_future is None
            self.call_soon(self._loop_self_reading)
            super().run_forever()
        finally:
            if self._self_reading_future is not None:
                ov = self._self_reading_future._ov
                self._self_reading_future.cancel()
                if ov is not None:
                    self._proactor._unregister(ov)
                self._self_reading_future = None

    async def create_pipe_connection(self, protocol_factory, address):
        f = self._proactor.connect_pipe(address)
        pipe = await f
        protocol = protocol_factory()
        trans = self._make_duplex_pipe_transport(pipe, protocol, extra={'addr': address})
        return (trans, protocol)

    async def start_serving_pipe(self, protocol_factory, address):
        server = PipeServer(address)

        def loop_accept_pipe(f=None):
            pipe = None
            try:
                if f:
                    pipe = f.result()
                    server._free_instances.discard(pipe)
                    if server.closed():
                        pipe.close()
                        return
                    protocol = protocol_factory()
                    self._make_duplex_pipe_transport(pipe, protocol, extra={'addr': address})
                pipe = server._get_unconnected_pipe()
                if pipe is None:
                    return
                f = self._proactor.accept_pipe(pipe)
            except BrokenPipeError:
                if pipe and pipe.fileno() != -1:
                    pipe.close()
                self.call_soon(loop_accept_pipe)
            except OSError as exc:
                if pipe and pipe.fileno() != -1:
                    self.call_exception_handler({'message': 'Pipe accept failed', 'exception': exc, 'pipe': pipe})
                    pipe.close()
                elif self._debug:
                    logger.warning('Accept pipe failed on pipe %r', pipe, exc_info=True)
                self.call_soon(loop_accept_pipe)
            except exceptions.CancelledError:
                if pipe:
                    pipe.close()
            else:
                server._accept_pipe_future = f
                f.add_done_callback(loop_accept_pipe)
        self.call_soon(loop_accept_pipe)
        return [server]

    async def _make_subprocess_transport(self, protocol, args, shell, stdin, stdout, stderr, bufsize, extra=None, **kwargs):
        waiter = self.create_future()
        transp = _WindowsSubprocessTransport(self, protocol, args, shell, stdin, stdout, stderr, bufsize, waiter=waiter, extra=extra, **kwargs)
        try:
            await waiter
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException:
            transp.close()
            await transp._wait()
            raise
        return transp