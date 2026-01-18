from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
class JointGrid(_BaseGrid):
    """Grid for drawing a bivariate plot with marginal univariate plots.

    Many plots can be drawn by using the figure-level interface :func:`jointplot`.
    Use this class directly when you need more flexibility.

    """

    def __init__(self, data=None, *, x=None, y=None, hue=None, height=6, ratio=5, space=0.2, palette=None, hue_order=None, hue_norm=None, dropna=False, xlim=None, ylim=None, marginal_ticks=False):
        f = plt.figure(figsize=(height, height))
        gs = plt.GridSpec(ratio + 1, ratio + 1)
        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
        self._figure = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)
        if not marginal_ticks:
            plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_x.get_yticklabels(), visible=False)
            plt.setp(ax_marg_y.get_xticklabels(), visible=False)
            plt.setp(ax_marg_x.get_yticklabels(minor=True), visible=False)
            plt.setp(ax_marg_y.get_xticklabels(minor=True), visible=False)
            ax_marg_x.yaxis.grid(False)
            ax_marg_y.xaxis.grid(False)
        p = VectorPlotter(data=data, variables=dict(x=x, y=y, hue=hue))
        plot_data = p.plot_data.loc[:, p.plot_data.notna().any()]
        if dropna:
            plot_data = plot_data.dropna()

        def get_var(var):
            vector = plot_data.get(var, None)
            if vector is not None:
                vector = vector.rename(p.variables.get(var, None))
            return vector
        self.x = get_var('x')
        self.y = get_var('y')
        self.hue = get_var('hue')
        for axis in 'xy':
            name = p.variables.get(axis, None)
            if name is not None:
                getattr(ax_joint, f'set_{axis}label')(name)
        if xlim is not None:
            ax_joint.set_xlim(xlim)
        if ylim is not None:
            ax_joint.set_ylim(ylim)
        self._hue_params = dict(palette=palette, hue_order=hue_order, hue_norm=hue_norm)
        utils.despine(f)
        if not marginal_ticks:
            utils.despine(ax=ax_marg_x, left=True)
            utils.despine(ax=ax_marg_y, bottom=True)
        for axes in [ax_marg_x, ax_marg_y]:
            for axis in [axes.xaxis, axes.yaxis]:
                axis.label.set_visible(False)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)

    def _inject_kwargs(self, func, kws, params):
        """Add params to kws if they are accepted by func."""
        func_params = signature(func).parameters
        for key, val in params.items():
            if key in func_params:
                kws.setdefault(key, val)

    def plot(self, joint_func, marginal_func, **kwargs):
        """Draw the plot by passing functions for joint and marginal axes.

        This method passes the ``kwargs`` dictionary to both functions. If you
        need more control, call :meth:`JointGrid.plot_joint` and
        :meth:`JointGrid.plot_marginals` directly with specific parameters.

        Parameters
        ----------
        joint_func, marginal_func : callables
            Functions to draw the bivariate and univariate plots. See methods
            referenced above for information about the required characteristics
            of these functions.
        kwargs
            Additional keyword arguments are passed to both functions.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        self.plot_marginals(marginal_func, **kwargs)
        self.plot_joint(joint_func, **kwargs)
        return self

    def plot_joint(self, func, **kwargs):
        """Draw a bivariate plot on the joint axes of the grid.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should accept ``x`` and ``y``. Otherwise,
            it must accept ``x`` and ``y`` vectors of data as the first two
            positional arguments, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, the function must
            accept ``hue`` as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            kwargs['ax'] = self.ax_joint
        else:
            plt.sca(self.ax_joint)
        if self.hue is not None:
            kwargs['hue'] = self.hue
            self._inject_kwargs(func, kwargs, self._hue_params)
        if str(func.__module__).startswith('seaborn'):
            func(x=self.x, y=self.y, **kwargs)
        else:
            func(self.x, self.y, **kwargs)
        return self

    def plot_marginals(self, func, **kwargs):
        """Draw univariate plots on each marginal axes.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should  accept ``x`` and ``y`` and plot
            when only one of them is defined. Otherwise, it must accept a vector
            of data as the first positional argument and determine its orientation
            using the ``vertical`` parameter, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, it must accept ``hue``
            as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        seaborn_func = str(func.__module__).startswith('seaborn') and (not func.__name__ == 'distplot')
        func_params = signature(func).parameters
        kwargs = kwargs.copy()
        if self.hue is not None:
            kwargs['hue'] = self.hue
            self._inject_kwargs(func, kwargs, self._hue_params)
        if 'legend' in func_params:
            kwargs.setdefault('legend', False)
        if 'orientation' in func_params:
            orient_kw_x = {'orientation': 'vertical'}
            orient_kw_y = {'orientation': 'horizontal'}
        elif 'vertical' in func_params:
            orient_kw_x = {'vertical': False}
            orient_kw_y = {'vertical': True}
        if seaborn_func:
            func(x=self.x, ax=self.ax_marg_x, **kwargs)
        else:
            plt.sca(self.ax_marg_x)
            func(self.x, **orient_kw_x, **kwargs)
        if seaborn_func:
            func(y=self.y, ax=self.ax_marg_y, **kwargs)
        else:
            plt.sca(self.ax_marg_y)
            func(self.y, **orient_kw_y, **kwargs)
        self.ax_marg_x.yaxis.get_label().set_visible(False)
        self.ax_marg_y.xaxis.get_label().set_visible(False)
        return self

    def refline(self, *, x=None, y=None, joint=True, marginal=True, color='.5', linestyle='--', **line_kws):
        """Add a reference line(s) to joint and/or marginal axes.

        Parameters
        ----------
        x, y : numeric
            Value(s) to draw the line(s) at.
        joint, marginal : bools
            Whether to add the reference line(s) to the joint/marginal axes.
        color : :mod:`matplotlib color <matplotlib.colors>`
            Specifies the color of the reference line(s).
        linestyle : str
            Specifies the style of the reference line(s).
        line_kws : key, value mappings
            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`
            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``
            is not None.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        line_kws['color'] = color
        line_kws['linestyle'] = linestyle
        if x is not None:
            if joint:
                self.ax_joint.axvline(x, **line_kws)
            if marginal:
                self.ax_marg_x.axvline(x, **line_kws)
        if y is not None:
            if joint:
                self.ax_joint.axhline(y, **line_kws)
            if marginal:
                self.ax_marg_y.axhline(y, **line_kws)
        return self

    def set_axis_labels(self, xlabel='', ylabel='', **kwargs):
        """Set axis labels on the bivariate axes.

        Parameters
        ----------
        xlabel, ylabel : strings
            Label names for the x and y variables.
        kwargs : key, value mappings
            Other keyword arguments are passed to the following functions:

            - :meth:`matplotlib.axes.Axes.set_xlabel`
            - :meth:`matplotlib.axes.Axes.set_ylabel`

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        self.ax_joint.set_xlabel(xlabel, **kwargs)
        self.ax_joint.set_ylabel(ylabel, **kwargs)
        return self