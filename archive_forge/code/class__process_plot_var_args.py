from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
class _process_plot_var_args:
    """
    Process variable length arguments to `~.Axes.plot`, to support ::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
    """

    def __init__(self, command='plot'):
        self.command = command
        self.set_prop_cycle(None)

    def set_prop_cycle(self, cycler):
        if cycler is None:
            cycler = mpl.rcParams['axes.prop_cycle']
        self._idx = 0
        self._cycler_items = [*cycler]
        self._prop_keys = cycler.keys

    def __call__(self, axes, *args, data=None, **kwargs):
        axes._process_unit_info(kwargs=kwargs)
        for pos_only in 'xy':
            if pos_only in kwargs:
                raise _api.kwarg_error(self.command, pos_only)
        if not args:
            return
        if data is None:
            args = [cbook.sanitize_sequence(a) for a in args]
        else:
            replaced = [mpl._replacer(data, arg) for arg in args]
            if len(args) == 1:
                label_namer_idx = 0
            elif len(args) == 2:
                try:
                    _process_plot_format(args[1])
                except ValueError:
                    label_namer_idx = 1
                else:
                    if replaced[1] is not args[1]:
                        _api.warn_external(f"Second argument {args[1]!r} is ambiguous: could be a format string but is in 'data'; using as data.  If it was intended as data, set the format string to an empty string to suppress this warning.  If it was intended as a format string, explicitly pass the x-values as well.  Alternatively, rename the entry in 'data'.", RuntimeWarning)
                        label_namer_idx = 1
                    else:
                        label_namer_idx = 0
            elif len(args) == 3:
                label_namer_idx = 1
            else:
                raise ValueError('Using arbitrary long args with data is not supported due to ambiguity of arguments; use multiple plotting calls instead')
            if kwargs.get('label') is None:
                kwargs['label'] = mpl._label_from_arg(replaced[label_namer_idx], args[label_namer_idx])
            args = replaced
        ambiguous_fmt_datakey = data is not None and len(args) == 2
        if len(args) >= 4 and (not cbook.is_scalar_or_string(kwargs.get('label'))):
            raise ValueError('plot() with multiple groups of data (i.e., pairs of x and y) does not support multiple labels')
        while args:
            this, args = (args[:2], args[2:])
            if args and isinstance(args[0], str):
                this += (args[0],)
                args = args[1:]
            yield from self._plot_args(axes, this, kwargs, ambiguous_fmt_datakey=ambiguous_fmt_datakey)

    def get_next_color(self):
        """Return the next color in the cycle."""
        if 'color' not in self._prop_keys:
            return 'k'
        c = self._cycler_items[self._idx]['color']
        self._idx = (self._idx + 1) % len(self._cycler_items)
        return c

    def _getdefaults(self, ignore, kw):
        """
        If some keys in the property cycle (excluding those in the set
        *ignore*) are absent or set to None in the dict *kw*, return a copy
        of the next entry in the property cycle, excluding keys in *ignore*.
        Otherwise, don't advance the property cycle, and return an empty dict.
        """
        prop_keys = self._prop_keys - ignore
        if any((kw.get(k, None) is None for k in prop_keys)):
            default_dict = self._cycler_items[self._idx].copy()
            self._idx = (self._idx + 1) % len(self._cycler_items)
            for p in ignore:
                default_dict.pop(p, None)
        else:
            default_dict = {}
        return default_dict

    def _setdefaults(self, defaults, kw):
        """
        Add to the dict *kw* the entries in the dict *default* that are absent
        or set to None in *kw*.
        """
        for k in defaults:
            if kw.get(k, None) is None:
                kw[k] = defaults[k]

    def _makeline(self, axes, x, y, kw, kwargs):
        kw = {**kw, **kwargs}
        default_dict = self._getdefaults(set(), kw)
        self._setdefaults(default_dict, kw)
        seg = mlines.Line2D(x, y, **kw)
        return (seg, kw)

    def _makefill(self, axes, x, y, kw, kwargs):
        x = axes.convert_xunits(x)
        y = axes.convert_yunits(y)
        kw = kw.copy()
        kwargs = kwargs.copy()
        ignores = {'marker', 'markersize', 'markeredgecolor', 'markerfacecolor', 'markeredgewidth'}
        for k, v in kwargs.items():
            if v is not None:
                ignores.add(k)
        default_dict = self._getdefaults(ignores, kw)
        self._setdefaults(default_dict, kw)
        facecolor = kw.get('color', None)
        default_dict.pop('color', None)
        self._setdefaults(default_dict, kwargs)
        seg = mpatches.Polygon(np.column_stack((x, y)), facecolor=facecolor, fill=kwargs.get('fill', True), closed=kw['closed'])
        seg.set(**kwargs)
        return (seg, kwargs)

    def _plot_args(self, axes, tup, kwargs, *, return_kwargs=False, ambiguous_fmt_datakey=False):
        """
        Process the arguments of ``plot([x], y, [fmt], **kwargs)`` calls.

        This processes a single set of ([x], y, [fmt]) parameters; i.e. for
        ``plot(x, y, x2, y2)`` it will be called twice. Once for (x, y) and
        once for (x2, y2).

        x and y may be 2D and thus can still represent multiple datasets.

        For multiple datasets, if the keyword argument *label* is a list, this
        will unpack the list and assign the individual labels to the datasets.

        Parameters
        ----------
        tup : tuple
            A tuple of the positional parameters. This can be one of

            - (y,)
            - (x, y)
            - (y, fmt)
            - (x, y, fmt)

        kwargs : dict
            The keyword arguments passed to ``plot()``.

        return_kwargs : bool
            Whether to also return the effective keyword arguments after label
            unpacking as well.

        ambiguous_fmt_datakey : bool
            Whether the format string in *tup* could also have been a
            misspelled data key.

        Returns
        -------
        result
            If *return_kwargs* is false, a list of Artists representing the
            dataset(s).
            If *return_kwargs* is true, a list of (Artist, effective_kwargs)
            representing the dataset(s). See *return_kwargs*.
            The Artist is either `.Line2D` (if called from ``plot()``) or
            `.Polygon` otherwise.
        """
        if len(tup) > 1 and isinstance(tup[-1], str):
            *xy, fmt = tup
            linestyle, marker, color = _process_plot_format(fmt, ambiguous_fmt_datakey=ambiguous_fmt_datakey)
        elif len(tup) == 3:
            raise ValueError('third arg must be a format string')
        else:
            xy = tup
            linestyle, marker, color = (None, None, None)
        if any((v is None for v in tup)):
            raise ValueError('x, y, and format string must not be None')
        kw = {}
        for prop_name, val in zip(('linestyle', 'marker', 'color'), (linestyle, marker, color)):
            if val is not None:
                if fmt.lower() != 'none' and prop_name in kwargs and (val != 'None'):
                    _api.warn_external(f'''{prop_name} is redundantly defined by the '{prop_name}' keyword argument and the fmt string "{fmt}" (-> {prop_name}={val!r}). The keyword argument will take precedence.''')
                kw[prop_name] = val
        if len(xy) == 2:
            x = _check_1d(xy[0])
            y = _check_1d(xy[1])
        else:
            x, y = index_of(xy[-1])
        if axes.xaxis is not None:
            axes.xaxis.update_units(x)
        if axes.yaxis is not None:
            axes.yaxis.update_units(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'x and y must have same first dimension, but have shapes {x.shape} and {y.shape}')
        if x.ndim > 2 or y.ndim > 2:
            raise ValueError(f'x and y can be no greater than 2D, but have shapes {x.shape} and {y.shape}')
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if self.command == 'plot':
            make_artist = self._makeline
        else:
            kw['closed'] = kwargs.get('closed', True)
            make_artist = self._makefill
        ncx, ncy = (x.shape[1], y.shape[1])
        if ncx > 1 and ncy > 1 and (ncx != ncy):
            raise ValueError(f'x has {ncx} columns but y has {ncy} columns')
        if ncx == 0 or ncy == 0:
            return []
        label = kwargs.get('label')
        n_datasets = max(ncx, ncy)
        if n_datasets > 1 and (not cbook.is_scalar_or_string(label)):
            if len(label) != n_datasets:
                raise ValueError(f'label must be scalar or have the same length as the input data, but found {len(label)} for {n_datasets} datasets.')
            labels = label
        else:
            labels = [label] * n_datasets
        result = (make_artist(axes, x[:, j % ncx], y[:, j % ncy], kw, {**kwargs, 'label': label}) for j, label in enumerate(labels))
        if return_kwargs:
            return list(result)
        else:
            return [l[0] for l in result]