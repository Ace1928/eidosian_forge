import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
def _get_nccl_collective_communicator(self, comm_key, device_list):
    """Create or retrieve an NCCL communicator from cache.

        If the communicator is found in cache, return the communicator. If not,
        a communicator and a stream will be created and put in cache.
        TODO(Hao): this function is not thread-safe now.

        Args:
            comm_key: the key to query the communicator cache.
            device_list: a list of GPU devices of the current process
                                that participates into the collective.

        Returns:
            communicator: the NCCL communicator corresponded to the devices.
        """
    if not comm_key:
        raise RuntimeError('Got empty communicator key.')
    for d in device_list:
        self._used_gpu_indices.add(d)
    if comm_key in self._dev_comm_map:
        return self._dev_comm_map[comm_key]
    group_key = self._generate_group_key(comm_key)
    if self.rank == 0:
        nccl_uid = self._generate_nccl_uid(group_key)
    else:
        rendezvous = Rendezvous(group_key)
        rendezvous.meet()
        nccl_uid = rendezvous.get_nccl_id()
    actual_world_size = len(device_list) * self.world_size
    comms = [None] * len(device_list)
    streams = [None] * len(device_list)
    events = [None] * len(device_list)
    nccl_util.groupStart()
    for i, device in enumerate(device_list):
        actual_rank = self.rank * len(device_list) + i
        with nccl_util.Device(device):
            comms[i] = nccl_util.create_nccl_communicator(actual_world_size, nccl_uid, actual_rank)
            streams[i] = get_stream_pool(device).get_stream()
            events[i] = cupy.cuda.Event()
    nccl_util.groupEnd()
    self._dev_comm_map[comm_key] = comms
    self._dev_streams_map[comm_key] = streams
    self._dev_event_map[comm_key] = events
    return comms