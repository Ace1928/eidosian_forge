import io
from typing import Iterator, List, Optional
import torch
from torch import Tensor
from torio.io._streaming_media_decoder import _get_afilter_desc, StreamingMediaDecoder as StreamReader
from torio.io._streaming_media_encoder import CodecConfig, StreamingMediaEncoder as StreamWriter
def _get_muxer(dtype: torch.dtype):
    types = {torch.uint8: 'u8', torch.int16: 's16le', torch.int32: 's32le', torch.float32: 'f32le', torch.float64: 'f64le'}
    if dtype not in types:
        raise ValueError(f'Unsupported dtype is provided {dtype}. Supported dtypes are: {types.keys()}')
    return types[dtype]