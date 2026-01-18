import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
class MemoryProfileTimeline:

    def __init__(self, memory_profile):
        """The minimum representation of the memory profile timeline
        includes the memory timeline and categories. The timeline
        consists of [timestamp, action, (TensorKey, version), numbytes]
        elements, to denote any actions (pre-existing, create, destroy,
        or increment_version) that occurred to a specific Tensor for a
        chunk of memory. The categories help map each (TensorKey,
        version) pair into a category."""
        self.timeline = memory_profile.timeline
        self.categories = memory_profile._categories

    def _coalesce_timeline(self, device_str):
        """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
        device = torch.device(device_str)
        times: List[int] = []
        sizes: List[List[int]] = []

        def update(key, version, delta):
            category = self.categories.get(key, version) if isinstance(key, TensorKey) else None
            index = _CATEGORY_TO_INDEX[category] + 1
            sizes[-1][index] += int(delta)
        t_min = -1
        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue
            if t != -1:
                t = int(t / 1000)
            if t_min == -1 or (t < t_min and t > 0):
                t_min = t
            if len(times) == 0:
                times.append(t)
                sizes.append([0] + [0 for _ in _CATEGORY_TO_INDEX])
            elif t != times[-1]:
                times.append(t)
                sizes.append(sizes[-1].copy())
            if action in (Action.PREEXISTING, Action.CREATE):
                update(key, version, numbytes)
            elif action == Action.INCREMENT_VERSION:
                update(key, version, -numbytes)
                update(key, version + 1, numbytes)
            elif action == Action.DESTROY:
                update(key, version, -numbytes)
            else:
                raise ValueError(f'Unknown action: {action}')
        times = [t_min if t < 0 else t for t in times]
        return (times, sizes)

    def export_memory_timeline(self, path, device) -> None:
        """Saves the memory timeline as [times, sizes by category]
        as a JSON formatted file to the given path for the given
        device."""
        times, sizes = self._coalesce_timeline(device)
        import json
        with open(path, 'w') as f:
            json.dump([times, sizes], f)

    def export_memory_timeline_raw(self, path, device_str) -> None:
        """Saves the memory timeline as raw memory event tuples in the
        form of (timestamp, action, numbytes, category)
        as a JSON formatted file to the given path for the given
        device."""
        device = torch.device(device_str)
        raw_events: List[Tuple[int, int, int, int]] = []

        def get_category_index(key, version):
            category = self.categories.get(key, version) if isinstance(key, TensorKey) else None
            return _CATEGORY_TO_INDEX[category]
        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue
            if action in (Action.PREEXISTING, Action.CREATE):
                raw_events.append((t, _ACTION_TO_INDEX[action], numbytes, get_category_index(key, version)))
            elif action == Action.INCREMENT_VERSION:
                raw_events.append((t, _ACTION_TO_INDEX[action], -numbytes, get_category_index(key, version)))
                raw_events.append((t, _ACTION_TO_INDEX[action], numbytes, get_category_index(key, version + 1)))
            elif action == Action.DESTROY:
                raw_events.append((t, _ACTION_TO_INDEX[action], -numbytes, get_category_index(key, version)))
            else:
                raise ValueError(f'Unknown action: {action}')
        import json
        with open(path, 'w') as f:
            json.dump(raw_events, f)

    def export_memory_timeline_html(self, path, device, figsize=(20, 12), title=None) -> None:
        """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
        import importlib.util
        matplotlib_spec = importlib.util.find_spec('matplotlib')
        if matplotlib_spec is None:
            print('export_memory_timeline_html failed because matplotlib was not found.')
            return
        from base64 import b64encode
        from os import remove
        from tempfile import NamedTemporaryFile
        import matplotlib.pyplot as plt
        import numpy as np
        mt = self._coalesce_timeline(device)
        times, sizes = (np.array(mt[0]), np.array(mt[1]))
        t_min = min(times)
        times -= t_min
        stacked = np.cumsum(sizes, axis=1) / 1024 ** 3
        max_memory_allocated = torch.cuda.max_memory_allocated()
        max_memory_reserved = torch.cuda.max_memory_reserved()
        fig = plt.figure(figsize=figsize, dpi=80)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            i = _CATEGORY_TO_INDEX[category]
            axes.fill_between(times / 1000.0, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7)
        fig.legend(['Unknown' if i is None else i.name for i in _CATEGORY_TO_COLORS])
        axes.set_xlabel('Time (ms)')
        axes.set_ylabel('Memory (GB)')
        title = '\n\n'.join(([title] if title else []) + [f'Max memory allocated: {max_memory_allocated / 10 ** 9:.2f} GB \nMax memory reserved: {max_memory_reserved / 10 ** 9:.2f} GB'])
        axes.set_title(title)
        tmpfile = NamedTemporaryFile('wb', suffix='.png', delete=False)
        tmpfile.close()
        fig.savefig(tmpfile.name, format='png')
        with open(tmpfile.name, 'rb') as tmp:
            encoded = b64encode(tmp.read()).decode('utf-8')
            html = f"""<html>\n<head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>\n<body>\n  <img src='data:image/png;base64,{encoded}'>\n</body>\n</html>"""
            with open(path, 'w') as f:
                f.write(html)
        remove(tmpfile.name)