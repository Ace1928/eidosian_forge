import collections
import contextlib
import logging
import socket
import threading
from oslo_config import cfg
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
Delete the attachment no longer in use, and disconnect volume
        if necessary.

        :param client: Cinderclient object
        :param attachment_id: ID of the attachment between volume and host
        :param volume_id: ID of the volume to attach
        :param host: The host the volume was attached to
        :param conn: connector object
        :param connection_info: connection information of the volume we are
                                detaching
        :device: device used to write image

        