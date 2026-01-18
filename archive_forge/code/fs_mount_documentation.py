import collections
import contextlib
import logging
import os
import socket
import threading
from oslo_concurrency import processutils
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
Mark an attachment as no longer in use, and unmount its mountpoint
        if necessary.

        :param vol_name: The name of the volume on the remote filesystem.
        :param mountpoint: The directory where the filesystem is be
                           mounted on the local compute host.
        :param host: The host the volume was attached to.
        