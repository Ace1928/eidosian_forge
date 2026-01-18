import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
def _profile_to_snapshot(profile):
    import torch
    from torch.profiler._memory_profiler import Action, TensorKey
    from torch._C._profiler import _EventType
    memory_profile = profile._memory_profile()
    allocation_stacks = {}
    for event in memory_profile._op_tree.sorted_nodes:
        if event.tag == _EventType.Allocation:
            parent = event.parent
            python_parents = []
            while parent:
                if parent.tag in (_EventType.PyCall, _EventType.PyCCall):
                    python_parents.append(parent)
                parent = parent.parent
            key = TensorKey.from_allocation(event.extra_fields)
            if key and event.extra_fields.alloc_size > 0:
                allocation_stacks[key] = python_parents
    device_count = torch.cuda.device_count()
    snapshot = {'device_traces': [[] for _ in range(device_count + 1)], 'segments': [{'device': device, 'address': None, 'total_size': 0, 'stream': 0, 'blocks': []} for device in range(device_count + 1)]}

    def to_device(device):
        if device.type == 'cuda':
            return device.index
        else:
            return device_count

    def allocate(size, tensor_key, version, during_trace=True):
        device = to_device(tensor_key.device)
        addr = tensor_key.storage.ptr
        seg = snapshot['segments'][device]
        if seg['address'] is None or seg['address'] > addr:
            seg['address'] = addr
        seg['total_size'] = max(seg['total_size'], addr + size)
        category = memory_profile._categories.get(tensor_key, version)
        category = category.name.lower() if category is not None else 'unknown'
        stack = allocation_stacks.get(tensor_key, ())
        stack = [{'filename': 'none', 'line': 0, 'name': p.name} for p in stack]
        r = {'action': 'alloc', 'addr': addr, 'size': size, 'stream': 0, 'frames': stack, 'category': category}
        if during_trace:
            snapshot['device_traces'][device].append(r)
        return r

    def free(alloc, device):
        for e in ('free_requested', 'free_completed'):
            snapshot['device_traces'][device].append({'action': e, 'addr': alloc['addr'], 'size': alloc['size'], 'stream': 0, 'frames': alloc['frames']})
    kv_to_elem = {}
    for time, action, (tensor_key, version), size in memory_profile.timeline:
        if not isinstance(tensor_key, TensorKey):
            continue
        if action == Action.CREATE:
            kv_to_elem[tensor_key, version] = allocate(size, tensor_key, version)
        elif action == Action.DESTROY:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))
        elif action == Action.INCREMENT_VERSION:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))
            kv_to_elem[tensor_key, version + 1] = allocate(size, tensor_key, version + 1)
        elif action == Action.PREEXISTING:
            kv_to_elem[tensor_key, version] = allocate(size, tensor_key, version, during_trace=False)
    blocks_at_end = [(to_device(tensor_key.device), event['addr'], event['size'], event['frames']) for (tensor_key, version), event in kv_to_elem.items()]
    for device, blocks in groupby(sorted(blocks_at_end), key=lambda x: x[0]):
        seg = snapshot['segments'][device]
        last_addr = seg['address']
        for _, addr, size, frames in blocks:
            if last_addr < addr:
                seg['blocks'].append({'size': addr - last_addr, 'state': 'inactive'})
            seg['blocks'].append({'size': size, 'state': 'active_allocated', 'requested_size': size, 'frames': frames})
            last_addr = addr + size
        if last_addr < seg['total_size']:
            seg['blocks'].append({'size': seg['total_size'] - last_addr, 'state': 'inactive'})
    snapshot['segments'] = [seg for seg in snapshot['segments'] if seg['blocks']]
    for seg in snapshot['segments']:
        seg['total_size'] -= seg['address']
        if not seg['blocks']:
            seg['blocks'].append({'size': seg['total_size'], 'state': 'inactive'})
    return snapshot