from __future__ import annotations
from itertools import product
import warnings
import numpy as np
import matplotlib.cm as mcm
import matplotlib.axes as mplaxes
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt
from . import core
from . import util
from .util.deprecation import rename_kw, Deprecated
from .util.exceptions import ParameterError
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Callable, Dict
from ._typing import _FloatLike_co
def __decorate_axis(axis, ax_type, key='C:maj', Sa=None, mela=None, thaat=None, unicode=True, fmin=None, unison=None, intervals=None, bins_per_octave=None, n_bins=None):
    """Configure axis tickers, locators, and labels"""
    time_units = {'h': 'hours', 'm': 'minutes', 's': 'seconds', 'ms': 'milliseconds'}
    if ax_type == 'tonnetz':
        axis.set_major_formatter(TonnetzFormatter())
        axis.set_major_locator(mplticker.FixedLocator(np.arange(6)))
        axis.set_label_text('Tonnetz')
    elif ax_type == 'chroma':
        axis.set_major_formatter(ChromaFormatter(key=key, unicode=unicode))
        degrees = core.key_to_degrees(key)
        axis.set_major_locator(mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel()))
        axis.set_label_text('Pitch class')
    elif ax_type == 'chroma_h':
        if Sa is None:
            Sa = 0
        axis.set_major_formatter(ChromaSvaraFormatter(Sa=Sa, unicode=unicode))
        if thaat is None:
            degrees = np.arange(12)
        else:
            degrees = core.thaat_to_degrees(thaat)
        degrees = np.mod(degrees + Sa, 12)
        axis.set_major_locator(mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel()))
        axis.set_label_text('Svara')
    elif ax_type == 'chroma_c':
        if Sa is None:
            Sa = 0
        axis.set_major_formatter(ChromaSvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
        degrees = core.mela_to_degrees(mela)
        degrees = np.mod(degrees + Sa, 12)
        axis.set_major_locator(mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel()))
        axis.set_label_text('Svara')
    elif ax_type == 'chroma_fjs':
        if fmin is None:
            fmin = core.note_to_hz('C1')
        if unison is None:
            unison = core.hz_to_note(fmin, octave=False, cents=False)
        axis.set_major_formatter(ChromaFJSFormatter(intervals=intervals, unison=unison, unicode=unicode, bins_per_octave=bins_per_octave))
        if isinstance(intervals, str) and bins_per_octave > 7:
            tick_intervals = core.interval_frequencies(7, fmin=1, intervals=intervals, bins_per_octave=bins_per_octave, sort=False)
            all_intervals = core.interval_frequencies(bins_per_octave, fmin=1, intervals=intervals, bins_per_octave=bins_per_octave, sort=True)
            degrees = util.match_events(tick_intervals, all_intervals)
        else:
            degrees = np.arange(bins_per_octave)
        axis.set_major_locator(mplticker.FixedLocator(degrees))
        axis.set_label_text('Pitch class')
    elif ax_type in ['tempo', 'fourier_tempo']:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        axis.set_major_locator(mplticker.LogLocator(base=2.0))
        axis.set_label_text('BPM')
    elif ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(unit=None, lag=False))
        axis.set_major_locator(mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time')
    elif ax_type in time_units:
        axis.set_major_formatter(TimeFormatter(unit=ax_type, lag=False))
        axis.set_major_locator(mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time ({:s})'.format(time_units[ax_type]))
    elif ax_type == 'lag':
        axis.set_major_formatter(TimeFormatter(unit=None, lag=True))
        axis.set_major_locator(mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Lag')
    elif isinstance(ax_type, str) and ax_type.startswith('lag_'):
        unit = ax_type[4:]
        axis.set_major_formatter(TimeFormatter(unit=unit, lag=True))
        axis.set_major_locator(mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Lag ({:s})'.format(time_units[unit]))
    elif ax_type == 'cqt_note':
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        log_C1 = np.log2(core.note_to_hz('C1'))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(C_offset,)))
        axis.set_minor_formatter(NoteFormatter(key=key, major=False, unicode=unicode))
        axis.set_minor_locator(mplticker.LogLocator(base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0)))
        axis.set_label_text('Note')
    elif ax_type == 'cqt_svara':
        axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
        sa_offset = 2.0 ** (np.log2(Sa) - np.floor(np.log2(Sa)))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(sa_offset,)))
        axis.set_minor_formatter(SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode))
        axis.set_minor_locator(mplticker.LogLocator(base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0)))
        axis.set_label_text('Svara')
    elif ax_type == 'vqt_fjs':
        if fmin is None:
            fmin = core.note_to_hz('C1')
        axis.set_major_formatter(FJSFormatter(intervals=intervals, fmin=fmin, unison=unison, unicode=unicode, bins_per_octave=bins_per_octave, n_bins=n_bins))
        log_fmin = np.log2(fmin)
        fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
        axis.set_minor_formatter(FJSFormatter(intervals=intervals, fmin=fmin, unison=unison, unicode=unicode, bins_per_octave=bins_per_octave, n_bins=n_bins, major=False))
        axis.set_minor_locator(mplticker.FixedLocator(core.interval_frequencies(n_bins * 12 // bins_per_octave, fmin=fmin, intervals=intervals, bins_per_octave=12)))
        axis.set_label_text('Note')
    elif ax_type == 'vqt_hz':
        if fmin is None:
            fmin = core.note_to_hz('C1')
        axis.set_major_formatter(LogHzFormatter())
        log_fmin = np.log2(fmin)
        fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(mplticker.LogLocator(base=2.0, subs=core.interval_frequencies(12, fmin=fmin_offset, intervals=intervals, bins_per_octave=12)))
        axis.set_label_text('Hz')
    elif ax_type == 'vqt_note':
        if fmin is None:
            fmin = core.note_to_hz('C1')
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        log_fmin = np.log2(fmin)
        fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
        axis.set_minor_formatter(NoteFormatter(key=key, unicode=unicode, major=False))
        axis.set_minor_locator(mplticker.LogLocator(base=2.0, subs=core.interval_frequencies(12, fmin=fmin_offset, intervals=intervals, bins_per_octave=12)))
        axis.set_label_text('Note')
    elif ax_type in ['cqt_hz']:
        axis.set_major_formatter(LogHzFormatter())
        log_C1 = np.log2(core.note_to_hz('C1'))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(C_offset,)))
        axis.set_major_locator(mplticker.LogLocator(base=2.0))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(mplticker.LogLocator(base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0)))
        axis.set_label_text('Hz')
    elif ax_type == 'fft_note':
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        log_C1 = np.log2(core.note_to_hz('C1'))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.SymmetricalLogLocator(axis.get_transform()))
        axis.set_minor_formatter(NoteFormatter(key=key, major=False, unicode=unicode))
        axis.set_minor_locator(mplticker.LogLocator(base=2.0, subs=2.0 ** (np.arange(1, 12) / 12.0)))
        axis.set_label_text('Note')
    elif ax_type == 'fft_svara':
        axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
        log_Sa = np.log2(Sa)
        sa_offset = 2.0 ** (log_Sa - np.floor(log_Sa))
        axis.set_major_locator(mplticker.SymmetricalLogLocator(axis.get_transform(), base=2.0, subs=[sa_offset]))
        axis.set_minor_formatter(SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode))
        axis.set_minor_locator(mplticker.LogLocator(base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0)))
        axis.set_label_text('Svara')
    elif ax_type in ['mel', 'log']:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        axis.set_major_locator(mplticker.SymmetricalLogLocator(axis.get_transform()))
        axis.set_label_text('Hz')
    elif ax_type in ['linear', 'hz', 'fft']:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        axis.set_label_text('Hz')
    elif ax_type in ['frames']:
        axis.set_label_text('Frames')
    elif ax_type in ['off', 'none', None]:
        axis.set_label_text('')
        axis.set_ticks([])
    else:
        raise ParameterError(f'Unsupported axis type: {ax_type}')