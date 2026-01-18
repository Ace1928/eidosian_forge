import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
class ConcreteProp(torch.fx.Interpreter):

    def __init__(self, mod, *, writer=None, skip_offload=False):
        super().__init__(mod)
        self.writer = writer
        self.skip_offload = skip_offload
        self.seen_storages = set()

    def run_node(self, n):
        self.pbar.update(1)
        r = super().run_node(n)
        name = n.name
        if isinstance(r, torch.Tensor):
            if self.writer is None:
                n.meta['concrete_value'] = r
            elif StorageWeakRef(r.untyped_storage()) in self.seen_storages:
                n.meta['concrete_value'] = None
            else:
                if not self.skip_offload:
                    self.writer.write_tensor(os.path.join('eager', name), r)
                n.meta['concrete_value'] = LoadTensorMeta(r.size(), r.stride(), r.dtype, r.device)
                self.seen_storages.add(StorageWeakRef(r.untyped_storage()))
        else:
            n.meta['concrete_value'] = is_tuple
        return r

    def propagate(self, *args):
        with tqdm(desc='Saving intermediates for delta debugging', total=len(self.module.graph.nodes), disable=self.writer is None) as pbar:
            self.pbar = pbar
            r = super().run(*args)
            if not self.skip_offload:
                pbar.set_description('Saved!  To skip next time, run with --skip-saving-eager-intermediates')
            return r