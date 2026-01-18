import errno
import logging
import os
from oslo_config import cfg
Delete cached file if present.

    :param cache: dictionary to hold opaque cache.
    :param filename: filename to delete
    