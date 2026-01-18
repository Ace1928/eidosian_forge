import io
import logging
import math
import re
import urllib
import eventlet
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _
import glance_store.location
class UploadPart(object):
    """The class for the upload part."""

    def __init__(self, mpu, fp, partnum, chunks):
        self.mpu = mpu
        self.partnum = partnum
        self.fp = fp
        self.size = 0
        self.chunks = chunks
        self.etag = {}
        self.success = True