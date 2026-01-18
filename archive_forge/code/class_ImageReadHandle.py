import logging
import ssl
import time
from oslo_utils import excutils
from oslo_utils import netutils
import requests
import urllib.parse as urlparse
from urllib3 import connection as httplib
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
class ImageReadHandle(object):
    """Read handle for glance images."""

    def __init__(self, glance_read_iter):
        """Initializes the read handle with given parameters.

        :param glance_read_iter: iterator to read data from glance image
        """
        self._glance_read_iter = glance_read_iter
        self._iter = self.get_next()

    def read(self, chunk_size):
        """Read an item from the image data iterator.

        The input chunk size is ignored since the client ImageBodyIterator
        uses its own chunk size.
        """
        try:
            data = next(self._iter)
            return data
        except StopIteration:
            LOG.debug('Completed reading data from the image iterator.')
            return ''

    def get_next(self):
        """Get the next item from the image iterator."""
        for data in self._glance_read_iter:
            yield data

    def close(self):
        """Close the read handle.

        This is a NOP.
        """
        pass

    def __str__(self):
        return 'Image read handle'