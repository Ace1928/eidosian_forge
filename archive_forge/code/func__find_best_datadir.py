import errno
import logging
import os
import stat
import urllib
import jsonschema
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LW
import glance_store.location
from the filesystem store. The users running the services that are
def _find_best_datadir(self, image_size):
    """Finds the best datadir by priority and free space.

        Traverse directories returning the first one that has sufficient
        free space, in priority order. If two suitable directories have
        the same priority, choose the one with the most free space
        available.
        :param image_size: size of image being uploaded.
        :returns: best_datadir as directory path of the best priority datadir.
        :raises: exceptions.StorageFull if there is no datadir in
                self.priority_data_map that can accommodate the image.
        """
    if not self.multiple_datadirs:
        return self.datadir
    best_datadir = None
    max_free_space = 0
    for priority in self.priority_list:
        for datadir in self.priority_data_map.get(priority):
            free_space = self._get_capacity_info(datadir)
            if free_space >= image_size and free_space > max_free_space:
                max_free_space = free_space
                best_datadir = datadir
        if best_datadir:
            break
    else:
        msg = _('There is no enough disk space left on the image storage media. requested=%s') % image_size
        LOG.exception(msg)
        raise exceptions.StorageFull(message=msg)
    return best_datadir