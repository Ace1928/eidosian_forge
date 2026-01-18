from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class _TimeSeries:
    """Metadata about time series data for a particular run and tag.

    Superclass of `ScalarTimeSeries`, `TensorTimeSeries`, and
    `BlobSequenceTimeSeries`.
    """
    __slots__ = ('_max_step', '_max_wall_time', '_plugin_content', '_description', '_display_name')

    def __init__(self, *, max_step, max_wall_time, plugin_content, description, display_name):
        self._max_step = max_step
        self._max_wall_time = max_wall_time
        self._plugin_content = plugin_content
        self._description = description
        self._display_name = display_name

    @property
    def max_step(self):
        return self._max_step

    @property
    def max_wall_time(self):
        return self._max_wall_time

    @property
    def plugin_content(self):
        return self._plugin_content

    @property
    def description(self):
        return self._description

    @property
    def display_name(self):
        return self._display_name