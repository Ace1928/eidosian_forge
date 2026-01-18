import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import _log_api_usage_once
from . import _video_opt
def _read_from_stream(container: 'av.container.Container', start_offset: float, end_offset: float, pts_unit: str, stream: 'av.stream.Stream', stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]]) -> List['av.frame.Frame']:
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()
    if pts_unit == 'sec':
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float('inf'):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")
    frames = {}
    should_buffer = True
    max_buffer_size = 5
    if stream.type == 'video':
        extradata = stream.codec_context.extradata
        if extradata and b'DivX' in extradata:
            pos = extradata.find(b'DivX')
            d = extradata[pos:]
            o = re.search(b'DivX(\\d+)Build(\\d+)(\\w)', d)
            if o is None:
                o = re.search(b'DivX(\\d+)b(\\d+)(\\w)', d)
            if o is not None:
                should_buffer = o.group(3) == b'p'
    seek_offset = start_offset
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        return []
    buffer_count = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames[frame.pts] = frame
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError:
        pass
    result = [frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset]
    if len(frames) > 0 and start_offset > 0 and (start_offset not in frames):
        preceding_frames = [i for i in frames if i < start_offset]
        if len(preceding_frames) > 0:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result