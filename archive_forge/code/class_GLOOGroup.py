import datetime
import logging
import os
import shutil
import time
import numpy
import pygloo
import ray
from ray._private import ray_constants
from ray.util.collective.collective_group import gloo_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
class GLOOGroup(BaseGroup):

    def __init__(self, world_size, rank, group_name, store_type='ray_internal_kv', device_type='tcp'):
        """Init an GLOO collective group.

        Args:
            world_size: The number of processes.
            rank: The id of process
            group_name: The unique user-specified group name.
            store_type: The store type. Optional: "redis",
                              "file", "hash".
            device_type: The device type to transport.
                               Optional: "tcp", "uv".
        """
        super(GLOOGroup, self).__init__(world_size, rank, group_name)
        self._gloo_context = gloo_util.create_gloo_context(self.rank, self.world_size)
        self._rendezvous = Rendezvous(self.group_name, self._gloo_context, store_type, device_type)
        self._rendezvous.meet()

    def destroy_group(self):
        """Destroy the group and release GLOO communicators."""
        self._rendezvous.destroy()
        if self._gloo_context is not None:
            pygloo.barrier(self._gloo_context)
            self._gloo_context = None
        if self.rank == 0 and self._rendezvous.store_type == 'file':
            store_name = get_store_name(self._group_name)
            store_path = gloo_util.get_gloo_store_path(store_name)
            if os.path.exists(store_path):
                shutil.rmtree(store_path)
        super(GLOOGroup, self).destroy_group()

    @classmethod
    def backend(cls):
        return Backend.GLOO

    def allreduce(self, tensors, allreduce_options=AllReduceOptions()):
        """AllReduce a list of tensors following options.

        Args:
            tensor: the tensor to be reduced, each tensor locates on CPU
            allreduce_options:

        Returns:
            None
        """

        def collective_fn(input_tensor, output_tensor, context):
            pygloo.allreduce(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor), gloo_util.get_gloo_reduce_op(allreduce_options.reduceOp))
        self._collective(tensors, tensors, collective_fn)

    def barrier(self, barrier_options=BarrierOptions()):
        """Blocks until all processes reach this barrier.

        Args:
            barrier_options: barrier options.

        Returns:
            None
        """
        barrier_tensor = numpy.array([1])
        self.allreduce([barrier_tensor])

    def reduce(self, tensors, reduce_options=ReduceOptions()):
        """Reduce tensors following options.

        Args:
            tensors: the list of tensors to be reduced,
                            this list only have one tensor.
            reduce_options: reduce options.

        Returns:
            None
        """
        root_rank = reduce_options.root_rank

        def collective_fn(input_tensor, output_tensor, context):
            pygloo.reduce(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor), gloo_util.get_gloo_reduce_op(reduce_options.reduceOp), root_rank)
        self._collective(tensors, tensors, collective_fn)

    def broadcast(self, tensors, broadcast_options=BroadcastOptions()):
        """Broadcast tensors to all other processes following options.

        Args:
            tensors: tensors to be broadcast or received.
            broadcast_options: broadcast options.

        Returns:
            None
        """
        root_rank = broadcast_options.root_rank

        def collective_fn(input_tensor, output_tensor, context):
            pygloo.broadcast(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor), root_rank)
        self._collective(tensors, tensors, collective_fn)

    def allgather(self, tensor_lists, tensors, allgather_options=AllGatherOptions()):
        """Allgather tensors on CPU into a list of tensors.

        Args:
            tensor_lists (List[List[Tensor]]): allgathered tensors.
            tensors: the list of tensors to allgather across the group.
                     Each tensor must locate on CPU.
            allgather_options: allgather options.

        Returns:
            None
        """

        def collective_fn(input_tensor, output_tensor, context):
            pygloo.allgather(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor))
        _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists)
        output_flattened = [_flatten_for_scatter_gather(tensor_list, copy=False) for tensor_list in tensor_lists]

        def postprocess_fn():
            for i, tensor_list in enumerate(tensor_lists):
                for j, tensor in enumerate(tensor_list):
                    gloo_util.copy_tensor(tensor, output_flattened[i][j])
        self._collective(tensors, output_flattened, collective_fn, postprocess_fn=postprocess_fn)

    def reducescatter(self, tensors, tensor_lists, reducescatter_options=ReduceScatterOptions()):
        """Reduce the scatter a list of tensors across the group.

        Args:
            tensors: the output tensors (could be unspecified), each
                            located on CPU.
            tensor_lists (List[List]): the list of tensors to be reduced then
                                       scattered.
            reducescatter_options: reduce-scatter options.

        Returns:
            None
        """

        def collective_fn(input_tensor, output_tensor, context):
            size = gloo_util.get_tensor_n_elements(input_tensor)
            world_size = self._gloo_context.size
            pygloo.reduce_scatter(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), size, [size // world_size for _ in range(world_size)], gloo_util.get_gloo_tensor_dtype(output_tensor), gloo_util.get_gloo_reduce_op(reducescatter_options.reduceOp))
        _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists)
        input_flattened = [_flatten_for_scatter_gather(tensor_list, copy=False) for tensor_list in tensor_lists]

        def preprocess_fn():
            for i, tensor_list in enumerate(tensor_lists):
                for j, tensor in enumerate(tensor_list):
                    gloo_util.copy_tensor(input_flattened[i][j], tensor)
        self._collective(input_flattened, tensors, collective_fn, preprocess_fn=preprocess_fn)

    def send(self, tensors, send_options=SendOptions()):
        """Send a tensor to a destination rank in the group.

        Args:
            tensors: the tensor to send.
            send_options: send options.

        Returns:
            None
        """

        def p2p_fn(tensor, context, peer):
            pygloo.send(context, gloo_util.get_tensor_ptr(tensor), gloo_util.get_tensor_n_elements(tensor), gloo_util.get_gloo_tensor_dtype(tensor), peer)
        self._point2point(tensors, p2p_fn, send_options.dst_rank)

    def recv(self, tensors, recv_options=RecvOptions()):
        """Receive a tensor from a source rank in the group.

        Args:
            tensors: the received tensor.
            recv_options: Receive options.

        Returns:
            None
        """

        def p2p_fn(tensor, context, peer):
            pygloo.recv(context, gloo_util.get_tensor_ptr(tensor), gloo_util.get_tensor_n_elements(tensor), gloo_util.get_gloo_tensor_dtype(tensor), peer)
        self._point2point(tensors, p2p_fn, recv_options.src_rank)

    def _collective(self, input_tensors, output_tensors, collective_fn, preprocess_fn=None, postprocess_fn=None):
        """A method to encapsulate all collective calls.

        Args:
            input_tensors: the list of the input tensors.
            output_tensors: the list of the output tensors.
            collective_fn: the collective function call.
            preprocess_fn: preprocess procedures before collective calls.
            postprocess_fn: postprocess procedures after collective calls.

        Returns:
            None
        """
        _check_cpu_tensors(input_tensors)
        _check_cpu_tensors(output_tensors)
        if preprocess_fn:
            preprocess_fn()
        collective_fn(input_tensors[0], output_tensors[0], self._gloo_context)
        if postprocess_fn:
            postprocess_fn()

    def _point2point(self, tensors, p2p_fn, peer_rank: int):
        """A method to encapsulate all peer-to-peer calls (i.e., send/recv).

        Args:
            tensors: the tensor to send or receive.
            p2p_fn: the p2p function call.
            peer_rank: the rank of the peer process.

        Returns:
            None
        """
        _check_cpu_tensors(tensors)
        p2p_fn(tensors[0], self._gloo_context, peer_rank)