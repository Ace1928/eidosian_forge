import asyncio
import atexit
import contextvars
import io
import os
import sys
import threading
import traceback
import warnings
from binascii import b2a_hex
from collections import defaultdict, deque
from io import StringIO, TextIOBase
from threading import local
from typing import Any, Callable, Deque, Dict, Optional
import zmq
from jupyter_client.session import extract_header
from tornado.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
class IOPubThread:
    """An object for sending IOPub messages in a background thread

    Prevents a blocking main thread from delaying output from threads.

    IOPubThread(pub_socket).background_socket is a Socket-API-providing object
    whose IO is always run in a thread.
    """

    def __init__(self, socket, pipe=False):
        """Create IOPub thread

        Parameters
        ----------
        socket : zmq.PUB Socket
            the socket on which messages will be sent.
        pipe : bool
            Whether this process should listen for IOPub messages
            piped from subprocesses.
        """
        self.socket = socket
        self._stopped = False
        self.background_socket = BackgroundSocket(self)
        self._master_pid = os.getpid()
        self._pipe_flag = pipe
        self.io_loop = IOLoop(make_current=False)
        if pipe:
            self._setup_pipe_in()
        self._local = threading.local()
        self._events: Deque[Callable[..., Any]] = deque()
        self._event_pipes: Dict[threading.Thread, Any] = {}
        self._event_pipe_gc_lock: threading.Lock = threading.Lock()
        self._event_pipe_gc_seconds: float = 10
        self._event_pipe_gc_task: Optional[asyncio.Task[Any]] = None
        self._setup_event_pipe()
        self.thread = threading.Thread(target=self._thread_main, name='IOPub')
        self.thread.daemon = True
        self.thread.pydev_do_not_trace = True
        self.thread.is_pydev_daemon_thread = True
        self.thread.name = 'IOPub'

    def _thread_main(self):
        """The inner loop that's actually run in a thread"""

        def _start_event_gc():
            self._event_pipe_gc_task = asyncio.ensure_future(self._run_event_pipe_gc())
        self.io_loop.run_sync(_start_event_gc)
        if not self._stopped:
            self.io_loop.start()
        if self._event_pipe_gc_task is not None:

            async def _cancel():
                self._event_pipe_gc_task.cancel()
            if not self._stopped:
                self.io_loop.run_sync(_cancel)
            else:
                self._event_pipe_gc_task.cancel()
        self.io_loop.close(all_fds=True)

    def _setup_event_pipe(self):
        """Create the PULL socket listening for events that should fire in this thread."""
        ctx = self.socket.context
        pipe_in = ctx.socket(zmq.PULL)
        pipe_in.linger = 0
        _uuid = b2a_hex(os.urandom(16)).decode('ascii')
        iface = self._event_interface = 'inproc://%s' % _uuid
        pipe_in.bind(iface)
        self._event_puller = ZMQStream(pipe_in, self.io_loop)
        self._event_puller.on_recv(self._handle_event)

    async def _run_event_pipe_gc(self):
        """Task to run event pipe gc continuously"""
        while True:
            await asyncio.sleep(self._event_pipe_gc_seconds)
            try:
                await self._event_pipe_gc()
            except Exception as e:
                print(f'Exception in IOPubThread._event_pipe_gc: {e}', file=sys.__stderr__)

    async def _event_pipe_gc(self):
        """run a single garbage collection on event pipes"""
        if not self._event_pipes:
            return
        with self._event_pipe_gc_lock:
            for thread, socket in list(self._event_pipes.items()):
                if not thread.is_alive():
                    socket.close()
                    del self._event_pipes[thread]

    @property
    def _event_pipe(self):
        """thread-local event pipe for signaling events that should be processed in the thread"""
        try:
            event_pipe = self._local.event_pipe
        except AttributeError:
            ctx = self.socket.context
            event_pipe = ctx.socket(zmq.PUSH)
            event_pipe.linger = 0
            event_pipe.connect(self._event_interface)
            self._local.event_pipe = event_pipe
            with self._event_pipe_gc_lock:
                self._event_pipes[threading.current_thread()] = event_pipe
        return event_pipe

    def _handle_event(self, msg):
        """Handle an event on the event pipe

        Content of the message is ignored.

        Whenever *an* event arrives on the event stream,
        *all* waiting events are processed in order.
        """
        n_events = len(self._events)
        for _ in range(n_events):
            event_f = self._events.popleft()
            event_f()

    def _setup_pipe_in(self):
        """setup listening pipe for IOPub from forked subprocesses"""
        ctx = self.socket.context
        self._pipe_uuid = os.urandom(16)
        pipe_in = ctx.socket(zmq.PULL)
        pipe_in.linger = 0
        try:
            self._pipe_port = pipe_in.bind_to_random_port('tcp://127.0.0.1')
        except zmq.ZMQError as e:
            warnings.warn("Couldn't bind IOPub Pipe to 127.0.0.1: %s" % e + '\nsubprocess output will be unavailable.', stacklevel=2)
            self._pipe_flag = False
            pipe_in.close()
            return
        self._pipe_in = ZMQStream(pipe_in, self.io_loop)
        self._pipe_in.on_recv(self._handle_pipe_msg)

    def _handle_pipe_msg(self, msg):
        """handle a pipe message from a subprocess"""
        if not self._pipe_flag or not self._is_master_process():
            return
        if msg[0] != self._pipe_uuid:
            print('Bad pipe message: %s', msg, file=sys.__stderr__)
            return
        self.send_multipart(msg[1:])

    def _setup_pipe_out(self):
        ctx = zmq.Context()
        pipe_out = ctx.socket(zmq.PUSH)
        pipe_out.linger = 3000
        pipe_out.connect('tcp://127.0.0.1:%i' % self._pipe_port)
        return (ctx, pipe_out)

    def _is_master_process(self):
        return os.getpid() == self._master_pid

    def _check_mp_mode(self):
        """check for forks, and switch to zmq pipeline if necessary"""
        if not self._pipe_flag or self._is_master_process():
            return MASTER
        return CHILD

    def start(self):
        """Start the IOPub thread"""
        self.thread.name = 'IOPub'
        self.thread.start()
        atexit.register(self.stop)

    def stop(self):
        """Stop the IOPub thread"""
        self._stopped = True
        if not self.thread.is_alive():
            return
        self.io_loop.add_callback(self.io_loop.stop)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            msg = 'IOPub thread did not terminate in 30 seconds'
            raise TimeoutError(msg)
        for _thread, event_pipe in self._event_pipes.items():
            event_pipe.close()

    def close(self):
        """Close the IOPub thread."""
        if self.closed:
            return
        self.socket.close()
        self.socket = None

    @property
    def closed(self):
        return self.socket is None

    def schedule(self, f):
        """Schedule a function to be called in our IO thread.

        If the thread is not running, call immediately.
        """
        if self.thread.is_alive():
            self._events.append(f)
            self._event_pipe.send(b'')
        else:
            f()

    def send_multipart(self, *args, **kwargs):
        """send_multipart schedules actual zmq send in my thread.

        If my thread isn't running (e.g. forked process), send immediately.
        """
        self.schedule(lambda: self._really_send(*args, **kwargs))

    def _really_send(self, msg, *args, **kwargs):
        """The callback that actually sends messages"""
        if self.closed:
            return
        mp_mode = self._check_mp_mode()
        if mp_mode != CHILD:
            self.socket.send_multipart(msg, *args, **kwargs)
        else:
            ctx, pipe_out = self._setup_pipe_out()
            pipe_out.send_multipart([self._pipe_uuid, *msg], *args, **kwargs)
            pipe_out.close()
            ctx.term()