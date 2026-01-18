import logging
import socket
import tempfile
from oslo_config import cfg
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _
Read up to a chunk's worth of data from the input stream into a
        file buffer.  Then return data out of that buffer.
        