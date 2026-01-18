import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
class GroupManager(object):
    """Use this class to manage the collective groups we created so far.

    Each process will have an instance of `GroupManager`. Each process
    could belong to multiple collective groups. The membership information
    and other metadata are stored in the global `_group_mgr` object.
    """

    def __init__(self):
        self._name_group_map = {}
        self._group_name_map = {}

    def create_collective_group(self, backend, world_size, rank, group_name):
        """The entry to create new collective groups in the manager.

        Put the registration and the group information into the manager
        metadata as well.
        """
        backend = types.Backend(backend)
        if backend == types.Backend.MPI:
            raise RuntimeError('Ray does not support MPI.')
        elif backend == types.Backend.GLOO:
            logger.debug("Creating GLOO group: '{}'...".format(group_name))
            g = GLOOGroup(world_size, rank, group_name, store_type='ray_internal_kv', device_type='tcp')
            self._name_group_map[group_name] = g
            self._group_name_map[g] = group_name
        elif backend == types.Backend.NCCL:
            logger.debug("Creating NCCL group: '{}'...".format(group_name))
            g = NCCLGroup(world_size, rank, group_name)
            self._name_group_map[group_name] = g
            self._group_name_map[g] = group_name
        return self._name_group_map[group_name]

    def is_group_exist(self, group_name):
        return group_name in self._name_group_map

    def get_group_by_name(self, group_name):
        """Get the collective group handle by its name."""
        if not self.is_group_exist(group_name):
            logger.warning("The group '{}' is not initialized.".format(group_name))
            return None
        return self._name_group_map[group_name]

    def destroy_collective_group(self, group_name):
        """Group destructor."""
        if not self.is_group_exist(group_name):
            logger.warning("The group '{}' does not exist.".format(group_name))
            return
        g = self._name_group_map[group_name]
        del self._group_name_map[g]
        del self._name_group_map[group_name]
        g.destroy_group()
        name = 'info_' + group_name
        try:
            store = ray.get_actor(name)
            ray.kill(store)
        except ValueError:
            pass