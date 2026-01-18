from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
class AbinitTimerSection:
    """Record with the timing results associated to a section of code."""
    STR_FIELDS = ('name',)
    NUMERIC_FIELDS = ('wall_time', 'wall_fract', 'cpu_time', 'cpu_fract', 'ncalls', 'gflops')
    FIELDS = tuple(STR_FIELDS + NUMERIC_FIELDS)

    @classmethod
    def fake(cls):
        """Return a fake section. Mainly used to fill missing entries if needed."""
        return AbinitTimerSection('fake', 0.0, 0.0, 0.0, 0.0, -1, 0.0)

    def __init__(self, name, cpu_time, cpu_fract, wall_time, wall_fract, ncalls, gflops):
        """
        Args:
            name: Name of the sections.
            cpu_time: CPU time in seconds.
            cpu_fract: Percentage of CPU time.
            wall_time: Wall-time in seconds.
            wall_fract: Percentage of wall-time.
            ncalls: Number of calls
            gflops: Gigaflops.
        """
        self.name = name.strip()
        self.cpu_time = float(cpu_time)
        self.cpu_fract = float(cpu_fract)
        self.wall_time = float(wall_time)
        self.wall_fract = float(wall_fract)
        self.ncalls = int(ncalls)
        self.gflops = float(gflops)

    def to_tuple(self):
        """Convert object to tuple."""
        return tuple((self.__dict__[at] for at in AbinitTimerSection.FIELDS))

    def to_dict(self):
        """Convert object to dictionary."""
        return {at: self.__dict__[at] for at in AbinitTimerSection.FIELDS}

    def to_csvline(self, with_header=False):
        """Return a string with data in CSV format. Add header if `with_header`."""
        string = ''
        if with_header:
            string += f'# {' '.join((at for at in AbinitTimerSection.FIELDS))}\n'
        string += ', '.join((str(v) for v in self.to_tuple())) + '\n'
        return string

    def __str__(self):
        """String representation."""
        string = ''
        for a in AbinitTimerSection.FIELDS:
            string = f'{a} = {self.__dict__[a]},'
        return string[:-1]