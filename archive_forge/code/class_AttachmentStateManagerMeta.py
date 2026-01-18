import collections
import contextlib
import logging
import socket
import threading
from oslo_config import cfg
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
class AttachmentStateManagerMeta(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super(AttachmentStateManagerMeta, cls).__call__(*args, **kwargs)
        return cls._instance[cls]