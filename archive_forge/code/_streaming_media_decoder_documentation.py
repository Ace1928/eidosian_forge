from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
Return an iterator that generates output tensors

        Arguments:
            timeout (float or None, optional): See
                :py:func:`~StreamingMediaDecoder.process_packet`. (Default: ``None``)

            backoff (float, optional): See
                :py:func:`~StreamingMediaDecoder.process_packet`. (Default: ``10.0``)

        Returns:
            Iterator[Tuple[Optional[ChunkTensor], ...]]:
                Iterator that yields a tuple of chunks that correspond to the output
                streams defined by client code.
                If an output stream is exhausted, then the chunk Tensor is substituted
                with ``None``.
                The iterator stops if all the output streams are exhausted.
        