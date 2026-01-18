import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCENodeImage(NodeImage):
    """A GCE Node Image class."""

    def __init__(self, id, name, driver, extra=None):
        super().__init__(id, name, driver, extra=extra)

    def delete(self):
        """
        Delete this image

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_delete_image(image=self)

    def deprecate(self, replacement, state, deprecated=None, obsolete=None, deleted=None):
        """
        Deprecate this image

        :param  replacement: Image to use as a replacement
        :type   replacement: ``str`` or :class: `GCENodeImage`

        :param  state: Deprecation state of this image. Possible values include
                       'ACTIVE', 'DELETED', 'DEPRECATED' or 'OBSOLETE'.
        :type   state: ``str``

        :param  deprecated: RFC3339 timestamp to mark DEPRECATED
        :type   deprecated: ``str`` or ``None``

        :param  obsolete: RFC3339 timestamp to mark OBSOLETE
        :type   obsolete: ``str`` or ``None``

        :param  deleted: RFC3339 timestamp to mark DELETED
        :type   deleted: ``str`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_deprecate_image(self, replacement, state, deprecated, obsolete, deleted)