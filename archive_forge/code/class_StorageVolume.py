import os
import re
import time
import atexit
import random
import socket
import hashlib
import binascii
import datetime
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Type, Tuple, Union, Callable, Optional
import libcloud.compute.ssh
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import b
from libcloud.common.base import BaseDriver, Connection, ConnectionKey
from libcloud.compute.ssh import SSHClient, BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.common.types import LibcloudError
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet, is_valid_ip_address
class StorageVolume(UuidMixin):
    """
    A base StorageVolume class to derive from.
    """

    def __init__(self, id, name, size, driver, state=None, extra=None):
        """
        :param id: Storage volume ID.
        :type id: ``str``

        :param name: Storage volume name.
        :type name: ``str``

        :param size: Size of this volume (in GB).
        :type size: ``int``

        :param driver: Driver this image belongs to.
        :type driver: :class:`.NodeDriver`

        :param state: Optional state of the StorageVolume. If not
                      provided, will default to UNKNOWN.
        :type state: :class:`.StorageVolumeState`

        :param extra: Optional provider specific attributes.
        :type extra: ``dict``
        """
        self.id = id
        self.name = name
        self.size = size
        self.driver = driver
        self.extra = extra
        self.state = state
        UuidMixin.__init__(self)

    def list_snapshots(self):
        """
        :rtype: ``list`` of ``VolumeSnapshot``
        """
        return self.driver.list_volume_snapshots(volume=self)

    def attach(self, node, device=None):
        """
        Attach this volume to a node.

        :param node: Node to attach volume to
        :type node: :class:`.Node`

        :param device: Where the device is exposed,
                            e.g. '/dev/sdb (optional)
        :type device: ``str``

        :return: ``True`` if attach was successful, ``False`` otherwise.
        :rtype: ``bool``
        """
        return self.driver.attach_volume(node=node, volume=self, device=device)

    def detach(self):
        """
        Detach this volume from its node

        :return: ``True`` if detach was successful, ``False`` otherwise.
        :rtype: ``bool``
        """
        return self.driver.detach_volume(volume=self)

    def snapshot(self, name):
        """
        Creates a snapshot of this volume.

        :return: Created snapshot.
        :rtype: ``VolumeSnapshot``
        """
        return self.driver.create_volume_snapshot(volume=self, name=name)

    def destroy(self):
        """
        Destroy this storage volume.

        :return: ``True`` if destroy was successful, ``False`` otherwise.
        :rtype: ``bool``
        """
        return self.driver.destroy_volume(volume=self)

    def __repr__(self):
        return '<StorageVolume id={} size={} driver={}>'.format(self.id, self.size, self.driver.name)