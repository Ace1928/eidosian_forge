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
class AbinitTimerParser(collections.abc.Iterable):
    """
    Responsible for parsing a list of output files, extracting the timing results
    and analyzing the results.
    Assume the Abinit output files have been produced with `timopt -1`.

    Example:
        parser = AbinitTimerParser()
        parser.parse(list_of_files)

    To analyze all *.abo files within top, use:

        parser, paths, okfiles = AbinitTimerParser.walk(top=".", ext=".abo")
    """
    BEGIN_TAG = '-<BEGIN_TIMER'
    END_TAG = '-<END_TIMER>'
    Error = AbinitTimerParseError

    @classmethod
    def walk(cls, top='.', ext='.abo'):
        """
        Scan directory tree starting from top, look for files with extension `ext` and
        parse timing data.

        Returns:
            parser: the new object
            paths: the list of files found
            ok_files: list of files that have been parsed successfully.
                (ok_files == paths) if all files have been parsed.
        """
        paths = []
        for root, _dirs, files in os.walk(top):
            for file in files:
                if file.endswith(ext):
                    paths.append(os.path.join(root, file))
        parser = cls()
        ok_files = parser.parse(paths)
        return (parser, paths, ok_files)

    def __init__(self):
        """Initialize object."""
        self._filenames = []
        self._timers = {}

    def __iter__(self):
        return iter(self._timers)

    def __len__(self):
        return len(self._timers)

    @property
    def filenames(self):
        """List of files that have been parsed successfully."""
        return self._filenames

    def parse(self, filenames):
        """
        Read and parse a filename or a list of filenames.
        Files that cannot be opened are ignored. A single filename may also be given.

        Returns:
            list of successfully read files.
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        read_ok = []
        for filename in filenames:
            try:
                file = open(filename)
            except OSError:
                logger.warning(f'Cannot open file {filename}')
                continue
            try:
                self._read(file, filename)
                read_ok.append(filename)
            except self.Error as exc:
                logger.warning(f'exception while parsing file {filename}:\n{exc}')
                continue
            finally:
                file.close()
        self._filenames.extend(read_ok)
        return read_ok

    def _read(self, fh, fname):
        """Parse the TIMER section."""
        if fname in self._timers:
            raise self.Error(f'Cannot overwrite timer associated to: {fname} ')

        def parse_line(line):
            """Parse single line."""
            name, vals = (line[:25], line[25:].split())
            try:
                ctime, cfract, wtime, wfract, ncalls, gflops = vals
            except ValueError:
                ctime, cfract, wtime, wfract, ncalls, gflops, _speedup, _eff = vals
            return AbinitTimerSection(name, ctime, cfract, wtime, wfract, ncalls, gflops)
        sections, info, cpu_time, wall_time = (None, None, None, None)
        data = {}
        parser_failed = False
        inside, has_timer = (0, False)
        for line in fh:
            if line.startswith(self.BEGIN_TAG):
                has_timer = True
                sections = []
                info = {}
                inside = 1
                line = line[len(self.BEGIN_TAG):].strip()[:-1]
                info['fname'] = fname
                for tok in line.split(','):
                    key, val = (s.strip() for s in tok.split('='))
                    info[key] = val
            elif line.startswith(self.END_TAG):
                inside = 0
                timer = AbinitTimer(sections, info, cpu_time, wall_time)
                mpi_rank = info['mpi_rank']
                data[mpi_rank] = timer
            elif inside:
                inside += 1
                line = line[1:].strip()
                if inside == 2:
                    dct = {}
                    for tok in line.split(','):
                        key, val = (s.strip() for s in tok.split('='))
                        dct[key] = float(val)
                    cpu_time, wall_time = (dct['cpu_time'], dct['wall_time'])
                elif inside > 5:
                    sections.append(parse_line(line))
                else:
                    try:
                        parse_line(line)
                    except Exception:
                        parser_failed = True
                    if not parser_failed:
                        raise self.Error(f'line should be empty: {inside}{line}')
        if not has_timer:
            raise self.Error(f'{fname}: No timer section found')
        self._timers[fname] = data

    def timers(self, filename=None, mpi_rank='0'):
        """Return the list of timers associated to the given `filename` and MPI rank mpi_rank."""
        if filename is not None:
            return [self._timers[filename][mpi_rank]]
        return [self._timers[filename][mpi_rank] for filename in self._filenames]

    def section_names(self, ordkey='wall_time'):
        """
        Return the names of sections ordered by ordkey.
        For the time being, the values are taken from the first timer.
        """
        section_names = []
        for idx, timer in enumerate(self.timers()):
            if idx == 0:
                section_names = [s.name for s in timer.order_sections(ordkey)]
        return section_names

    def get_sections(self, section_name):
        """
        Return the list of sections stored in self.timers() given `section_name`
        A fake section is returned if the timer does not have section_name.
        """
        sections = []
        for timer in self.timers():
            for sect in timer.sections:
                if sect.name == section_name:
                    sections.append(sect)
                    break
            else:
                sections.append(AbinitTimerSection.fake())
        return sections

    def pefficiency(self):
        """
        Analyze the parallel efficiency.

        Returns:
            ParallelEfficiency object.
        """
        timers = self.timers()
        ncpus = [timer.ncpus for timer in timers]
        min_idx = np.argmin(ncpus)
        min_ncpus = ncpus[min_idx]
        ref_t = timers[min_idx]
        peff = {}
        ctime_peff = [min_ncpus * ref_t.wall_time / (t.wall_time * ncp) for t, ncp in zip(timers, ncpus)]
        wtime_peff = [min_ncpus * ref_t.cpu_time / (t.cpu_time * ncp) for t, ncp in zip(timers, ncpus)]
        n = len(timers)
        peff['total'] = {}
        peff['total']['cpu_time'] = ctime_peff
        peff['total']['wall_time'] = wtime_peff
        peff['total']['cpu_fract'] = n * [100]
        peff['total']['wall_fract'] = n * [100]
        for sect_name in self.section_names():
            ref_sect = ref_t.get_section(sect_name)
            sects = [timer.get_section(sect_name) for timer in timers]
            try:
                ctime_peff = [min_ncpus * ref_sect.cpu_time / (s.cpu_time * ncp) for s, ncp in zip(sects, ncpus)]
                wtime_peff = [min_ncpus * ref_sect.wall_time / (s.wall_time * ncp) for s, ncp in zip(sects, ncpus)]
            except ZeroDivisionError:
                ctime_peff = n * [-1]
                wtime_peff = n * [-1]
            assert sect_name not in peff
            peff[sect_name] = {}
            peff[sect_name]['cpu_time'] = ctime_peff
            peff[sect_name]['wall_time'] = wtime_peff
            peff[sect_name]['cpu_fract'] = [s.cpu_fract for s in sects]
            peff[sect_name]['wall_fract'] = [s.wall_fract for s in sects]
        return ParallelEfficiency(self._filenames, min_idx, peff)

    def summarize(self, **kwargs):
        """Return pandas DataFrame with the most important results stored in the timers."""
        col_names = ['fname', 'wall_time', 'cpu_time', 'mpi_nprocs', 'omp_nthreads', 'mpi_rank']
        frame = pd.DataFrame(columns=col_names)
        for timer in self.timers():
            frame = frame.append({key: getattr(timer, key) for key in col_names}, ignore_index=True)
        frame['tot_ncpus'] = frame['mpi_nprocs'] * frame['omp_nthreads']
        idx = frame['tot_ncpus'].argmin()
        ref_wtime = frame.iloc[idx]['wall_time']
        ref_ncpus = frame.iloc[idx]['tot_ncpus']
        frame['peff'] = ref_ncpus * ref_wtime / (frame['wall_time'] * frame['tot_ncpus'])
        return frame

    @add_fig_kwargs
    def plot_efficiency(self, key='wall_time', what='good+bad', nmax=5, ax: plt.Axes=None, **kwargs):
        """
        Plot the parallel efficiency.

        Args:
            key: Parallel efficiency is computed using the wall_time.
            what: Specifies what to plot: `good` for sections with good parallel efficiency.
                `bad` for sections with bad efficiency. Options can be concatenated with `+`.
            nmax: Maximum number of entries in plot
            ax: matplotlib Axes or None if a new figure should be created.

        ================  ====================================================
        kwargs            Meaning
        ================  ====================================================
        linewidth         matplotlib linewidth. Default: 2.0
        markersize        matplotlib markersize. Default: 10
        ================  ====================================================

        Returns:
            `matplotlib` figure
        """
        ax, fig = get_ax_fig(ax=ax)
        lw = kwargs.pop('linewidth', 2.0)
        msize = kwargs.pop('markersize', 10)
        what = what.split('+')
        timers = self.timers()
        peff = self.pefficiency()
        n = len(timers)
        xx = np.arange(n)
        ax.set_prop_cycle(color=['g', 'b', 'c', 'm', 'y', 'k'])
        lines, legend_entries = ([], [])
        if 'good' in what:
            good = peff.good_sections(key=key, nmax=nmax)
            for g in good:
                yy = peff[g][key]
                line, = ax.plot(xx, yy, '-->', linewidth=lw, markersize=msize)
                lines.append(line)
                legend_entries.append(g)
        if 'bad' in what:
            bad = peff.bad_sections(key=key, nmax=nmax)
            for b in bad:
                yy = peff[b][key]
                line, = ax.plot(xx, yy, '-.<', linewidth=lw, markersize=msize)
                lines.append(line)
                legend_entries.append(b)
        if 'total' not in legend_entries:
            yy = peff['total'][key]
            total_line, = ax.plot(xx, yy, 'r', linewidth=lw, markersize=msize)
            lines.append(total_line)
            legend_entries.append('total')
        ax.legend(lines, legend_entries, loc='best', shadow=True)
        ax.set_xlabel('Total_NCPUs')
        ax.set_ylabel('Efficiency')
        ax.grid(visible=True)
        labels = [f'MPI={timer.mpi_nprocs}, OMP={timer.omp_nthreads}' for timer in timers]
        ax.set_xticks(xx)
        ax.set_xticklabels(labels, fontdict=None, minor=False, rotation=15)
        return fig

    @add_fig_kwargs
    def plot_pie(self, key='wall_time', minfract=0.05, **kwargs):
        """
        Plot pie charts of the different timers.

        Args:
            key: Keyword used to extract data from timers.
            minfract: Don't show sections whose relative weight is less that minfract.

        Returns:
            `matplotlib` figure
        """
        timers = self.timers()
        n = len(timers)
        fig = plt.gcf()
        gspec = GridSpec(n, 1)
        for idx, timer in enumerate(timers):
            ax = plt.subplot(gspec[idx, 0])
            ax.set_title(str(timer))
            timer.pie(ax=ax, key=key, minfract=minfract, show=False)
        return fig

    @add_fig_kwargs
    def plot_stacked_hist(self, key='wall_time', nmax=5, ax: plt.Axes=None, **kwargs):
        """
        Plot stacked histogram of the different timers.

        Args:
            key: Keyword used to extract data from the timers. Only the first `nmax`
                sections with largest value are show.
            nmax: Maximum number of sections to show. Other entries are grouped together
                in the `others` section.
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            `matplotlib` figure
        """
        ax, fig = get_ax_fig(ax=ax)
        mpi_rank = '0'
        timers = self.timers(mpi_rank=mpi_rank)
        n = len(timers)
        names, values = ([], [])
        rest = np.zeros(n)
        for idx, sec_name in enumerate(self.section_names(ordkey=key)):
            sections = self.get_sections(sec_name)
            sec_vals = np.asarray([s.__dict__[key] for s in sections])
            if idx < nmax:
                names.append(sec_name)
                values.append(sec_vals)
            else:
                rest += sec_vals
        names.append(f'others (nmax={nmax!r})')
        values.append(rest)
        ind = np.arange(n)
        width = 0.35
        colors = nmax * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
        bars = []
        bottom = np.zeros(n)
        for idx, vals in enumerate(values):
            color = colors[idx]
            bar_ = ax.bar(ind, vals, width, color=color, bottom=bottom)
            bars.append(bar_)
            bottom += vals
        ax.set_ylabel(key)
        ax.set_title(f'Stacked histogram with the {nmax} most important sections')
        ticks = ind + width / 2.0
        labels = [f'MPI={timer.mpi_nprocs}, OMP={timer.omp_nthreads}' for timer in timers]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend([bar_[0] for bar_ in bars], names, loc='best')
        return fig

    def plot_all(self, show=True, **kwargs):
        """Call all plot methods provided by the parser."""
        figs = []
        app = figs.append
        app(self.plot_stacked_hist(show=show))
        app(self.plot_efficiency(show=show))
        app(self.plot_pie(show=show))
        return figs