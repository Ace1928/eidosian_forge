from __future__ import annotations
import socket
import uuid
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty
from time import monotonic
from kombu.exceptions import ChannelError
from kombu.log import get_logger
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _get_or_create_session(self, queue):
    """Get or create consul session.

        Try to renew the session if it exists, otherwise create a new
        session in Consul.

        This session is used to acquire a lock inside Consul so that we achieve
        read-consistency between the nodes.

        Arguments:
        ---------
            queue (str): The name of the Queue.

        Returns
        -------
            str: The ID of the session.
        """
    try:
        session_id = self.queues[queue]['session_id']
    except KeyError:
        session_id = None
    return self._renew_existing_session(session_id) if session_id is not None else self._create_new_session()