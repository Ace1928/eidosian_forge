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
class AbinitTimer:
    """Container class storing the timing results."""

    def __init__(self, sections, info, cpu_time, wall_time):
        """
        Args:
            sections: List of sections
            info: Dictionary with extra info.
            cpu_time: Cpu-time in seconds.
            wall_time: Wall-time in seconds.
        """
        self.sections = tuple(sections)
        self.section_names = tuple((s.name for s in self.sections))
        self.info = info
        self.cpu_time = float(cpu_time)
        self.wall_time = float(wall_time)
        self.mpi_nprocs = int(info['mpi_nprocs'])
        self.omp_nthreads = int(info['omp_nthreads'])
        self.mpi_rank = info['mpi_rank'].strip()
        self.fname = info['fname'].strip()

    def __repr__(self):
        file, wall_time, mpi_nprocs, omp_nthreads = (self.fname, self.wall_time, self.mpi_nprocs, self.omp_nthreads)
        return f'{type(self).__name__}(file={file!r}, wall_time={wall_time:.3}, mpi_nprocs={mpi_nprocs!r}, omp_nthreads={omp_nthreads!r})'

    @property
    def ncpus(self):
        """Total number of CPUs employed."""
        return self.mpi_nprocs * self.omp_nthreads

    def get_section(self, section_name):
        """Return section associated to `section_name`."""
        try:
            idx = self.section_names.index(section_name)
        except Exception:
            raise
        sect = self.sections[idx]
        assert sect.name == section_name
        return sect

    def to_csv(self, fileobj=sys.stdout):
        """Write data on file fileobj using CSV format."""
        is_str = isinstance(fileobj, str)
        if is_str:
            fileobj = open(fileobj, mode='w')
        for idx, section in enumerate(self.sections):
            fileobj.write(section.to_csvline(with_header=idx == 0))
        fileobj.flush()
        if is_str:
            fileobj.close()

    def to_table(self, sort_key='wall_time', stop=None):
        """Return a table (list of lists) with timer data."""
        table = [list(AbinitTimerSection.FIELDS)]
        ord_sections = self.order_sections(sort_key)
        if stop is not None:
            ord_sections = ord_sections[:stop]
        for osect in ord_sections:
            row = list(map(str, osect.to_tuple()))
            table.append(row)
        return table
    totable = to_table

    def get_dataframe(self, sort_key='wall_time', **kwargs):
        """Return a pandas DataFrame with entries sorted according to `sort_key`."""
        frame = pd.DataFrame(columns=AbinitTimerSection.FIELDS)
        for osect in self.order_sections(sort_key):
            frame = frame.append(osect.to_dict(), ignore_index=True)
        frame.info = self.info
        frame.cpu_time = self.cpu_time
        frame.wall_time = self.wall_time
        frame.mpi_nprocs = self.mpi_nprocs
        frame.omp_nthreads = self.omp_nthreads
        frame.mpi_rank = self.mpi_rank
        frame.fname = self.fname
        return frame

    def get_values(self, keys):
        """Return a list of values associated to a particular list of keys."""
        if isinstance(keys, str):
            return [sec.__dict__[keys] for sec in self.sections]
        values = []
        for key in keys:
            values.append([sec.__dict__[key] for sec in self.sections])
        return values

    def names_and_values(self, key, minval=None, minfract=None, sorted=True):
        """
        Select the entries whose value[key] is >= minval or whose fraction[key] is >= minfract
        Return the names of the sections and the corresponding values.
        """
        values = self.get_values(key)
        names = self.get_values('name')
        new_names, new_values = ([], [])
        other_val = 0.0
        if minval is not None:
            assert minfract is None
            for name, val in zip(names, values):
                if val >= minval:
                    new_names.append(name)
                    new_values.append(val)
                else:
                    other_val += val
            new_names.append(f'below minval {minval}')
            new_values.append(other_val)
        elif minfract is not None:
            assert minval is None
            total = self.sum_sections(key)
            for name, val in zip(names, values):
                if val / total >= minfract:
                    new_names.append(name)
                    new_values.append(val)
                else:
                    other_val += val
            new_names.append(f'below minfract {minfract}')
            new_values.append(other_val)
        else:
            new_names, new_values = (names, values)
        if sorted:
            nandv = list(zip(new_names, new_values))
            nandv.sort(key=lambda t: t[1])
            new_names, new_values = ([n[0] for n in nandv], [n[1] for n in nandv])
        return (new_names, new_values)

    def _reduce_sections(self, keys, operator):
        return operator(self.get_values(keys))

    def sum_sections(self, keys):
        """Sum value of keys."""
        return self._reduce_sections(keys, sum)

    def order_sections(self, key, reverse=True):
        """Sort sections according to the value of key."""
        return sorted(self.sections, key=lambda s: s.__dict__[key], reverse=reverse)

    @add_fig_kwargs
    def cpuwall_histogram(self, ax: plt.Axes=None, **kwargs):
        """
        Plot histogram with cpu- and wall-time on axis `ax`.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            plt.Figure: matplotlib figure
        """
        ax, fig = get_ax_fig(ax=ax)
        ind = np.arange(len(self.sections))
        width = 0.35
        cpu_times = self.get_values('cpu_time')
        rects1 = plt.bar(ind, cpu_times, width, color='r')
        wall_times = self.get_values('wall_time')
        rects2 = plt.bar(ind + width, wall_times, width, color='y')
        ax.set_ylabel('Time (s)')
        ticks = self.get_values('name')
        ax.set_xticks(ind + width, ticks)
        ax.legend((rects1[0], rects2[0]), ('CPU', 'Wall'), loc='best')
        return fig

    @add_fig_kwargs
    def pie(self, key='wall_time', minfract=0.05, ax: plt.Axes=None, **kwargs):
        """
        Plot pie chart for this timer.

        Args:
            key: Keyword used to extract data from the timer.
            minfract: Don't show sections whose relative weight is less that minfract.
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            plt.Figure: matplotlib figure
        """
        ax, fig = get_ax_fig(ax=ax)
        ax.axis('equal')
        labels, vals = self.names_and_values(key, minfract=minfract)
        ax.pie(vals, explode=None, labels=labels, autopct='%1.1f%%', shadow=True)
        return fig

    @add_fig_kwargs
    def scatter_hist(self, ax: plt.Axes=None, **kwargs):
        """
        Scatter plot + histogram.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            plt.Figure: matplotlib figure
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        ax, fig = get_ax_fig(ax=ax)
        x = np.asarray(self.get_values('cpu_time'))
        y = np.asarray(self.get_values('wall_time'))
        axScatter = plt.subplot(1, 1, 1)
        axScatter.scatter(x, y)
        axScatter.set_aspect('auto')
        divider = make_axes_locatable(axScatter)
        axHistx = divider.append_axes('top', 1.2, pad=0.1, sharex=axScatter)
        axHisty = divider.append_axes('right', 1.2, pad=0.1, sharey=axScatter)
        plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)
        binwidth = 0.25
        xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
        lim = (int(xymax / binwidth) + 1) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)
        axHistx.hist(x, bins=bins)
        axHisty.hist(y, bins=bins, orientation='horizontal')
        axHistx.set_yticks([0, 50, 100])
        for tl in axHistx.get_xticklabels():
            tl.set_visible(False)
            for tl in axHisty.get_yticklabels():
                tl.set_visible(False)
                axHisty.set_xticks([0, 50, 100])
        return fig