from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@final
class TiledSequence:
    """Tiled sequence of chunks.

    Transform a sequence of stacked chunks to tiled chunks.

    Parameters:
        stackshape:
            Shape of stacked sequence excluding chunks.
        chunkshape:
            Shape of chunks.
        axestiled:
            Axes to be tiled. Map stacked sequence axis
            to chunk axis. By default, the sequence is not tiled.
        axes:
            Character codes for dimensions in stackshape and chunkshape.

    Examples:
        >>> ts = TiledSequence((1, 2), (3, 4), axestiled={1: 0}, axes='ABYX')
        >>> ts.shape
        (1, 6, 4)
        >>> ts.chunks
        (1, 3, 4)
        >>> ts.axes
        'AYX'

    """
    chunks: tuple[int, ...]
    'Shape of chunks in tiled sequence.'
    shape: tuple[int, ...]
    'Shape of tiled sequence including chunks.'
    axes: str | tuple[str, ...] | None
    'Dimensions codes of tiled sequence.'
    shape_squeezed: tuple[int, ...]
    'Shape of tiled sequence with length-1 dimensions removed.'
    axes_squeezed: str | tuple[str, ...] | None
    'Dimensions codes of tiled sequence with length-1 dimensions removed.'
    _stackdims: int
    'Number of dimensions in stack excluding chunks.'
    _chunkdims: int
    'Number of dimensions in chunks.'
    _shape_untiled: tuple[int, ...]
    'Shape of untiled sequence (stackshape + chunkshape).'
    _axestiled: tuple[tuple[int, int], ...]
    'Map axes to tile from stack to chunks.'

    def __init__(self, stackshape: Sequence[int], chunkshape: Sequence[int], /, *, axestiled: dict[int, int] | Sequence[tuple[int, int]] | None=None, axes: str | Sequence[str] | None=None) -> None:
        self._stackdims = len(stackshape)
        self._chunkdims = len(chunkshape)
        self._shape_untiled = tuple(stackshape) + tuple(chunkshape)
        if axes is not None and len(axes) != len(self._shape_untiled):
            raise ValueError('axes length does not match stackshape + chunkshape')
        if axestiled:
            axestiled = dict(axestiled)
            for ax0, ax1 in axestiled.items():
                axestiled[ax0] = ax1 + self._stackdims
            self._axestiled = tuple(reversed(sorted(axestiled.items())))
            axes_list = [] if axes is None else list(axes)
            shape = list(self._shape_untiled)
            chunks = [1] * self._stackdims + list(chunkshape)
            used = set()
            for ax0, ax1 in self._axestiled:
                if ax0 in used or ax1 in used:
                    raise ValueError('duplicate axis')
                used.add(ax0)
                used.add(ax1)
                shape[ax1] *= stackshape[ax0]
            for ax0, ax1 in self._axestiled:
                del shape[ax0]
                del chunks[ax0]
                if axes_list:
                    del axes_list[ax0]
            self.shape = tuple(shape)
            self.chunks = tuple(chunks)
            if axes is None:
                self.axes = None
            elif isinstance(axes, str):
                self.axes = ''.join(axes_list)
            else:
                self.axes = tuple(axes_list)
        else:
            self._axestiled = ()
            self.shape = self._shape_untiled
            self.chunks = (1,) * self._stackdims + tuple(chunkshape)
            if axes is None:
                self.axes = None
            elif isinstance(axes, str):
                self.axes = axes
            else:
                self.axes = tuple(axes)
        assert len(self.shape) == len(self.chunks)
        if self.axes is not None:
            assert len(self.shape) == len(self.axes)
        if self.axes is None:
            self.shape_squeezed = tuple((i for i in self.shape if i > 1))
            self.axes_squeezed = None
        else:
            keep = ('X', 'Y', 'width', 'length', 'height')
            self.shape_squeezed = tuple((i for i, ax in zip(self.shape, self.axes) if i > 1 or ax in keep))
            squeezed = tuple((ax for i, ax in zip(self.shape, self.axes) if i > 1 or ax in keep))
            self.axes_squeezed = ''.join(squeezed) if isinstance(self.axes, str) else squeezed

    def indices(self, indices: Iterable[Sequence[int]], /) -> Iterator[tuple[int, ...]]:
        """Return iterator over chunk indices of tiled sequence.

        Parameters:
            indices: Indices of chunks in stacked sequence.

        """
        chunkindex = [0] * self._chunkdims
        for index in indices:
            if index is None:
                yield None
            else:
                if len(index) != self._stackdims:
                    raise ValueError(f'{len(index)} != {self._stackdims}')
                index = list(index) + chunkindex
                for ax0, ax1 in self._axestiled:
                    index[ax1] = index[ax0]
                for ax0, ax1 in self._axestiled:
                    del index[ax0]
                yield tuple(index)

    def slices(self, indices: Iterable[Sequence[int]] | None=None, /) -> Iterator[tuple[int | slice, ...]]:
        """Return iterator over slices of chunks in tiled sequence.

        Parameters:
            indices: Indices of chunks in stacked sequence.

        """
        wholeslice: list[int | slice]
        chunkslice: list[int | slice] = [slice(None)] * self._chunkdims
        if indices is None:
            indices = numpy.ndindex(self._shape_untiled[:self._stackdims])
        for index in indices:
            if index is None:
                yield None
            else:
                assert len(index) == self._stackdims
                wholeslice = [*index, *chunkslice]
                for ax0, ax1 in self._axestiled:
                    j = self._shape_untiled[ax1]
                    i = cast(int, wholeslice[ax0]) * j
                    wholeslice[ax1] = slice(i, i + j)
                for ax0, ax1 in self._axestiled:
                    del wholeslice[ax0]
                yield tuple(wholeslice)

    @property
    def ndim(self) -> int:
        """Number of dimensions of tiled sequence excluding chunks."""
        return len(self.shape)

    @property
    def is_tiled(self) -> bool:
        """Sequence is tiled."""
        return bool(self._axestiled)