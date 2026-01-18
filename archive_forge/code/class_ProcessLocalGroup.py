import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
class ProcessLocalGroup(dist.ProcessGroup):
    _coll_lock = threading.Lock()
    _cur_coll_on_pgs = {}
    _terminate = threading.Event()

    @classmethod
    def _start_coll(cls, collective, pg):
        with cls._coll_lock:
            if pg.pg_name not in cls._cur_coll_on_pgs:
                cls._cur_coll_on_pgs[pg.pg_name] = Collective(pg.size(), collective, cls)
            return cls._cur_coll_on_pgs[pg.pg_name]

    @classmethod
    def _end_coll(cls, collective, pg):
        with cls._coll_lock:
            if pg.pg_name in cls._cur_coll_on_pgs and cls._cur_coll_on_pgs[pg.pg_name] == collective:
                cls._cur_coll_on_pgs.pop(pg.pg_name)

    @classmethod
    def exception_handle(cls, exc):
        cls._terminate.set()
        for coll in cls._cur_coll_on_pgs.values():
            with coll._start_cond:
                coll._start_cond.notify()
            with coll._done_cond:
                coll._done_cond.notify_all()

    @classmethod
    def reset(cls):
        with cls._coll_lock:
            cls._cur_coll_on_pgs = {}
            cls._terminate.clear()

    def alltoall(self, output_tensor_list, input_tensor_list, opts=AllToAllOptions()):
        coll = ProcessLocalGroup._start_coll(AllToAll(), self)
        res = coll.join(self._rank, (output_tensor_list, input_tensor_list))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        coll = ProcessLocalGroup._start_coll(AllReduce(opts.reduceOp), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        coll = ProcessLocalGroup._start_coll(AllReduce(opts.reduceOp), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def barrier(self, opts=BarrierOptions()):
        return self.allreduce(tensor_list=[torch.ones(1)])

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        coll = ProcessLocalGroup._start_coll(AllGather(), self)
        res = coll.join(self._rank, (output_tensors, input_tensor))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def _allgather_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        tensor_list = list(torch.chunk(output_tensor, self._world_size))
        return self.allgather([tensor_list], [input_tensor], opts)

    def broadcast(self, tensor_list, opts=BroadcastOptions()):
        coll = ProcessLocalGroup._start_coll(Broadcast(opts.rootRank), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def scatter(self, output_tensors, input_tensors, opts=ScatterOptions()):
        coll = ProcessLocalGroup._start_coll(Scatter(opts.rootRank), self)
        res = coll.join(self._rank, (output_tensors, input_tensors))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def gather(self, output_tensors, input_tensors, opts=ScatterOptions()):
        coll = ProcessLocalGroup._start_coll(Gather(opts.rootRank), self)
        res = coll.join(self._rank, (output_tensors, input_tensors))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        coll = ProcessLocalGroup._start_coll(ReduceScatter(opts.reduceOp), self)
        res = coll.join(self._rank, (output_tensor, scatter_list))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        tensor_list = list(torch.chunk(input_tensor, self._world_size))
        return self.reduce_scatter([output_tensor], [tensor_list], opts)

    def allgather_into_tensor_coalesced(self, output_tensor_list, input_tensor_list):
        res = None
        for o_t, i_t in zip(output_tensor_list, input_tensor_list):
            res = self._allgather_base(o_t, i_t)
        return res

    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size
        world = dist.distributed_c10d._world
        if isinstance(world, ThreadLocalWorld):
            world = world._get_world()
        self._world = weakref.ref(world)
        self._ctx = torch.autograd.set_multithreading_enabled(False)

    def size(self):
        return self._world_size

    @property
    def pg_name(self):
        """
        return the global registered name of the current pg in the world
        """
        return self._world().pg_names[self]

    def getBackendName(self):
        return 'threaded'

    def __repr__(self):
        return f'ThreadedPG world_size:{self._world_size} rank:{self._rank}'