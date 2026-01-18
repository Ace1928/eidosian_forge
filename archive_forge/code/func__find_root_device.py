from __future__ import annotations
import os
import tempfile
import typing
from typing import Any, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rbd as rbd_privsep
from os_brick import utils
def _find_root_device(self, connection_properties: dict[str, Any], conf) -> Optional[str]:
    """Find the underlying /dev/rbd* device for a mapping.

        Use the showmapped command to list all acive mappings and find the
        underlying /dev/rbd* device that corresponds to our pool and volume.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
        :type connection_properties: dict
        :returns: '/dev/rbd*' or None if no active mapping is found.
        """
    __, volume = connection_properties['name'].split('/')
    cmd = ['rbd', 'showmapped', '--format=json']
    cmd += self._get_rbd_args(connection_properties, conf)
    out, err = self._execute(*cmd, root_helper=self._root_helper, run_as_root=True)
    mappings = jsonutils.loads(out)
    if isinstance(mappings, dict):
        mappings = mappings.values()
    for mapping in mappings:
        if mapping['name'] == volume:
            return mapping['device']
    return None