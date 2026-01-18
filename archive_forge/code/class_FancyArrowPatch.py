import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
class FancyArrowPatch(Patch):
    """
    A fancy arrow patch.

    It draws an arrow using the `ArrowStyle`. It is primarily used by the
    `~.axes.Axes.annotate` method.  For most purposes, use the annotate method for
    drawing arrows.

    The head and tail positions are fixed at the specified start and end points
    of the arrow, but the size and shape (in display coordinates) of the arrow
    does not change when the axis is moved or zoomed.
    """
    _edge_default = True

    def __str__(self):
        if self._posA_posB is not None:
            (x1, y1), (x2, y2) = self._posA_posB
            return f'{type(self).__name__}(({x1:g}, {y1:g})->({x2:g}, {y2:g}))'
        else:
            return f'{type(self).__name__}({self._path_original})'

    @_docstring.dedent_interpd
    def __init__(self, posA=None, posB=None, *, path=None, arrowstyle='simple', connectionstyle='arc3', patchA=None, patchB=None, shrinkA=2, shrinkB=2, mutation_scale=1, mutation_aspect=1, **kwargs):
        """
        There are two ways for defining an arrow:

        - If *posA* and *posB* are given, a path connecting two points is
          created according to *connectionstyle*. The path will be
          clipped with *patchA* and *patchB* and further shrunken by
          *shrinkA* and *shrinkB*. An arrow is drawn along this
          resulting path using the *arrowstyle* parameter.

        - Alternatively if *path* is provided, an arrow is drawn along this
          path and *patchA*, *patchB*, *shrinkA*, and *shrinkB* are ignored.

        Parameters
        ----------
        posA, posB : (float, float), default: None
            (x, y) coordinates of arrow tail and arrow head respectively.

        path : `~matplotlib.path.Path`, default: None
            If provided, an arrow is drawn along this path and *patchA*,
            *patchB*, *shrinkA*, and *shrinkB* are ignored.

        arrowstyle : str or `.ArrowStyle`, default: 'simple'
            The `.ArrowStyle` with which the fancy arrow is drawn.  If a
            string, it should be one of the available arrowstyle names, with
            optional comma-separated attributes.  The optional attributes are
            meant to be scaled with the *mutation_scale*.  The following arrow
            styles are available:

            %(ArrowStyle:table)s

        connectionstyle : str or `.ConnectionStyle` or None, optional, default: 'arc3'
            The `.ConnectionStyle` with which *posA* and *posB* are connected.
            If a string, it should be one of the available connectionstyle
            names, with optional comma-separated attributes.  The following
            connection styles are available:

            %(ConnectionStyle:table)s

        patchA, patchB : `~matplotlib.patches.Patch`, default: None
            Head and tail patches, respectively.

        shrinkA, shrinkB : float, default: 2
            Shrinking factor of the tail and head of the arrow respectively.

        mutation_scale : float, default: 1
            Value with which attributes of *arrowstyle* (e.g., *head_length*)
            will be scaled.

        mutation_aspect : None or float, default: None
            The height of the rectangle will be squeezed by this value before
            the mutation and the mutated box will be stretched by the inverse
            of it.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties, optional
            Here is a list of available `.Patch` properties:

        %(Patch:kwdoc)s

            In contrast to other patches, the default ``capstyle`` and
            ``joinstyle`` for `FancyArrowPatch` are set to ``"round"``.
        """
        kwargs.setdefault('joinstyle', JoinStyle.round)
        kwargs.setdefault('capstyle', CapStyle.round)
        super().__init__(**kwargs)
        if posA is not None and posB is not None and (path is None):
            self._posA_posB = [posA, posB]
            if connectionstyle is None:
                connectionstyle = 'arc3'
            self.set_connectionstyle(connectionstyle)
        elif posA is None and posB is None and (path is not None):
            self._posA_posB = None
        else:
            raise ValueError('Either posA and posB, or path need to provided')
        self.patchA = patchA
        self.patchB = patchB
        self.shrinkA = shrinkA
        self.shrinkB = shrinkB
        self._path_original = path
        self.set_arrowstyle(arrowstyle)
        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect
        self._dpi_cor = 1.0

    def set_positions(self, posA, posB):
        """
        Set the start and end positions of the connecting path.

        Parameters
        ----------
        posA, posB : None, tuple
            (x, y) coordinates of arrow tail and arrow head respectively. If
            `None` use current value.
        """
        if posA is not None:
            self._posA_posB[0] = posA
        if posB is not None:
            self._posA_posB[1] = posB
        self.stale = True

    def set_patchA(self, patchA):
        """
        Set the tail patch.

        Parameters
        ----------
        patchA : `.patches.Patch`
        """
        self.patchA = patchA
        self.stale = True

    def set_patchB(self, patchB):
        """
        Set the head patch.

        Parameters
        ----------
        patchB : `.patches.Patch`
        """
        self.patchB = patchB
        self.stale = True

    @_docstring.dedent_interpd
    def set_connectionstyle(self, connectionstyle=None, **kwargs):
        """
        Set the connection style, possibly with further attributes.

        Attributes from the previous connection style are not reused.

        Without argument (or with ``connectionstyle=None``), the available box
        styles are returned as a human-readable string.

        Parameters
        ----------
        connectionstyle : str or `~matplotlib.patches.ConnectionStyle`
            The style of the connection: either a `.ConnectionStyle` instance,
            or a string, which is the style name and optionally comma separated
            attributes (e.g. "Arc,armA=30,rad=10"). Such a string is used to
            construct a `.ConnectionStyle` object, as documented in that class.

            The following connection styles are available:

            %(ConnectionStyle:table_and_accepts)s

        **kwargs
            Additional attributes for the connection style. See the table above
            for supported parameters.

        Examples
        --------
        ::

            set_connectionstyle("Arc,armA=30,rad=10")
            set_connectionstyle("arc", armA=30, rad=10)
        """
        if connectionstyle is None:
            return ConnectionStyle.pprint_styles()
        self._connector = ConnectionStyle(connectionstyle, **kwargs) if isinstance(connectionstyle, str) else connectionstyle
        self.stale = True

    def get_connectionstyle(self):
        """Return the `ConnectionStyle` used."""
        return self._connector

    def set_arrowstyle(self, arrowstyle=None, **kwargs):
        """
        Set the arrow style, possibly with further attributes.

        Attributes from the previous arrow style are not reused.

        Without argument (or with ``arrowstyle=None``), the available box
        styles are returned as a human-readable string.

        Parameters
        ----------
        arrowstyle : str or `~matplotlib.patches.ArrowStyle`
            The style of the arrow: either a `.ArrowStyle` instance, or a
            string, which is the style name and optionally comma separated
            attributes (e.g. "Fancy,head_length=0.2"). Such a string is used to
            construct a `.ArrowStyle` object, as documented in that class.

            The following arrow styles are available:

            %(ArrowStyle:table_and_accepts)s

        **kwargs
            Additional attributes for the arrow style. See the table above for
            supported parameters.

        Examples
        --------
        ::

            set_arrowstyle("Fancy,head_length=0.2")
            set_arrowstyle("fancy", head_length=0.2)
        """
        if arrowstyle is None:
            return ArrowStyle.pprint_styles()
        self._arrow_transmuter = ArrowStyle(arrowstyle, **kwargs) if isinstance(arrowstyle, str) else arrowstyle
        self.stale = True

    def get_arrowstyle(self):
        """Return the arrowstyle object."""
        return self._arrow_transmuter

    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        Parameters
        ----------
        scale : float
        """
        self._mutation_scale = scale
        self.stale = True

    def get_mutation_scale(self):
        """
        Return the mutation scale.

        Returns
        -------
        scalar
        """
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        Parameters
        ----------
        aspect : float
        """
        self._mutation_aspect = aspect
        self.stale = True

    def get_mutation_aspect(self):
        """Return the aspect ratio of the bbox mutation."""
        return self._mutation_aspect if self._mutation_aspect is not None else 1

    def get_path(self):
        """Return the path of the arrow in the data coordinates."""
        _path, fillable = self._get_path_in_displaycoord()
        if np.iterable(fillable):
            _path = Path.make_compound_path(*_path)
        return self.get_transform().inverted().transform_path(_path)

    def _get_path_in_displaycoord(self):
        """Return the mutated path of the arrow in display coordinates."""
        dpi_cor = self._dpi_cor
        if self._posA_posB is not None:
            posA = self._convert_xy_units(self._posA_posB[0])
            posB = self._convert_xy_units(self._posA_posB[1])
            posA, posB = self.get_transform().transform((posA, posB))
            _path = self.get_connectionstyle()(posA, posB, patchA=self.patchA, patchB=self.patchB, shrinkA=self.shrinkA * dpi_cor, shrinkB=self.shrinkB * dpi_cor)
        else:
            _path = self.get_transform().transform_path(self._path_original)
        _path, fillable = self.get_arrowstyle()(_path, self.get_mutation_scale() * dpi_cor, self.get_linewidth() * dpi_cor, self.get_mutation_aspect())
        return (_path, fillable)

    def draw(self, renderer):
        if not self.get_visible():
            return
        self._dpi_cor = renderer.points_to_pixels(1.0)
        path, fillable = self._get_path_in_displaycoord()
        if not np.iterable(fillable):
            path = [path]
            fillable = [fillable]
        affine = transforms.IdentityTransform()
        self._draw_paths_with_artist_properties(renderer, [(p, affine, self._facecolor if f and self._facecolor[3] else None) for p, f in zip(path, fillable)])