from __future__ import annotations
import logging
from copy import copy
import zmq
class PUBHandler(logging.Handler):
    """A basic logging handler that emits log messages through a PUB socket.

    Takes a PUB socket already bound to interfaces or an interface to bind to.

    Example::

        sock = context.socket(zmq.PUB)
        sock.bind('inproc://log')
        handler = PUBHandler(sock)

    Or::

        handler = PUBHandler('inproc://loc')

    These are equivalent.

    Log messages handled by this handler are broadcast with ZMQ topics
    ``this.root_topic`` comes first, followed by the log level
    (DEBUG,INFO,etc.), followed by any additional subtopics specified in the
    message by: log.debug("subtopic.subsub::the real message")
    """
    ctx: zmq.Context
    socket: zmq.Socket

    def __init__(self, interface_or_socket: str | zmq.Socket, context: zmq.Context | None=None, root_topic: str='') -> None:
        logging.Handler.__init__(self)
        self.root_topic = root_topic
        self.formatters = {logging.DEBUG: logging.Formatter('%(levelname)s %(filename)s:%(lineno)d - %(message)s\n'), logging.INFO: logging.Formatter('%(message)s\n'), logging.WARN: logging.Formatter('%(levelname)s %(filename)s:%(lineno)d - %(message)s\n'), logging.ERROR: logging.Formatter('%(levelname)s %(filename)s:%(lineno)d - %(message)s - %(exc_info)s\n'), logging.CRITICAL: logging.Formatter('%(levelname)s %(filename)s:%(lineno)d - %(message)s\n')}
        if isinstance(interface_or_socket, zmq.Socket):
            self.socket = interface_or_socket
            self.ctx = self.socket.context
        else:
            self.ctx = context or zmq.Context()
            self.socket = self.ctx.socket(zmq.PUB)
            self.socket.bind(interface_or_socket)

    @property
    def root_topic(self) -> str:
        return self._root_topic

    @root_topic.setter
    def root_topic(self, value: str):
        self.setRootTopic(value)

    def setRootTopic(self, root_topic: str):
        """Set the root topic for this handler.

        This value is prepended to all messages published by this handler, and it
        defaults to the empty string ''. When you subscribe to this socket, you must
        set your subscription to an empty string, or to at least the first letter of
        the binary representation of this string to ensure you receive any messages
        from this handler.

        If you use the default empty string root topic, messages will begin with
        the binary representation of the log level string (INFO, WARN, etc.).
        Note that ZMQ SUB sockets can have multiple subscriptions.
        """
        if isinstance(root_topic, bytes):
            root_topic = root_topic.decode('utf8')
        self._root_topic = root_topic

    def setFormatter(self, fmt, level=logging.NOTSET):
        """Set the Formatter for this handler.

        If no level is provided, the same format is used for all levels. This
        will overwrite all selective formatters set in the object constructor.
        """
        if level == logging.NOTSET:
            for fmt_level in self.formatters.keys():
                self.formatters[fmt_level] = fmt
        else:
            self.formatters[level] = fmt

    def format(self, record):
        """Format a record."""
        return self.formatters[record.levelno].format(record)

    def emit(self, record):
        """Emit a log message on my socket."""
        try:
            topic, msg = str(record.msg).split(TOPIC_DELIM, 1)
        except ValueError:
            topic = ''
        else:
            record = copy(record)
            record.msg = msg
        try:
            bmsg = self.format(record).encode('utf8')
        except Exception:
            self.handleError(record)
            return
        topic_list = []
        if self.root_topic:
            topic_list.append(self.root_topic)
        topic_list.append(record.levelname)
        if topic:
            topic_list.append(topic)
        btopic = '.'.join(topic_list).encode('utf8', 'replace')
        self.socket.send_multipart([btopic, bmsg])