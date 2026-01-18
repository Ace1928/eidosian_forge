import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
class SubplotSpec:
    """
    The location of a subplot in a `GridSpec`.

    .. note::

        Likely, you will never instantiate a `SubplotSpec` yourself. Instead,
        you will typically obtain one from a `GridSpec` using item-access.

    Parameters
    ----------
    gridspec : `~matplotlib.gridspec.GridSpec`
        The GridSpec, which the subplot is referencing.
    num1, num2 : int
        The subplot will occupy the *num1*-th cell of the given
        *gridspec*.  If *num2* is provided, the subplot will span between
        *num1*-th cell and *num2*-th cell **inclusive**.

        The index starts from 0.
    """

    def __init__(self, gridspec, num1, num2=None):
        self._gridspec = gridspec
        self.num1 = num1
        self.num2 = num2

    def __repr__(self):
        return f'{self.get_gridspec()}[{self.rowspan.start}:{self.rowspan.stop}, {self.colspan.start}:{self.colspan.stop}]'

    @staticmethod
    def _from_subplot_args(figure, args):
        """
        Construct a `.SubplotSpec` from a parent `.Figure` and either

        - a `.SubplotSpec` -- returned as is;
        - one or three numbers -- a MATLAB-style subplot specifier.
        """
        if len(args) == 1:
            arg, = args
            if isinstance(arg, SubplotSpec):
                return arg
            elif not isinstance(arg, Integral):
                raise ValueError(f'Single argument to subplot must be a three-digit integer, not {arg!r}')
            try:
                rows, cols, num = map(int, str(arg))
            except ValueError:
                raise ValueError(f'Single argument to subplot must be a three-digit integer, not {arg!r}') from None
        elif len(args) == 3:
            rows, cols, num = args
        else:
            raise _api.nargs_error('subplot', takes='1 or 3', given=len(args))
        gs = GridSpec._check_gridspec_exists(figure, rows, cols)
        if gs is None:
            gs = GridSpec(rows, cols, figure=figure)
        if isinstance(num, tuple) and len(num) == 2:
            if not all((isinstance(n, Integral) for n in num)):
                raise ValueError(f'Subplot specifier tuple must contain integers, not {num}')
            i, j = num
        else:
            if not isinstance(num, Integral) or num < 1 or num > rows * cols:
                raise ValueError(f'num must be an integer with 1 <= num <= {rows * cols}, not {num!r}')
            i = j = num
        return gs[i - 1:j]

    @property
    def num2(self):
        return self.num1 if self._num2 is None else self._num2

    @num2.setter
    def num2(self, value):
        self._num2 = value

    def get_gridspec(self):
        return self._gridspec

    def get_geometry(self):
        """
        Return the subplot geometry as tuple ``(n_rows, n_cols, start, stop)``.

        The indices *start* and *stop* define the range of the subplot within
        the `GridSpec`. *stop* is inclusive (i.e. for a single cell
        ``start == stop``).
        """
        rows, cols = self.get_gridspec().get_geometry()
        return (rows, cols, self.num1, self.num2)

    @property
    def rowspan(self):
        """The rows spanned by this subplot, as a `range` object."""
        ncols = self.get_gridspec().ncols
        return range(self.num1 // ncols, self.num2 // ncols + 1)

    @property
    def colspan(self):
        """The columns spanned by this subplot, as a `range` object."""
        ncols = self.get_gridspec().ncols
        c1, c2 = sorted([self.num1 % ncols, self.num2 % ncols])
        return range(c1, c2 + 1)

    def is_first_row(self):
        return self.rowspan.start == 0

    def is_last_row(self):
        return self.rowspan.stop == self.get_gridspec().nrows

    def is_first_col(self):
        return self.colspan.start == 0

    def is_last_col(self):
        return self.colspan.stop == self.get_gridspec().ncols

    def get_position(self, figure):
        """
        Update the subplot position from ``figure.subplotpars``.
        """
        gridspec = self.get_gridspec()
        nrows, ncols = gridspec.get_geometry()
        rows, cols = np.unravel_index([self.num1, self.num2], (nrows, ncols))
        fig_bottoms, fig_tops, fig_lefts, fig_rights = gridspec.get_grid_positions(figure)
        fig_bottom = fig_bottoms[rows].min()
        fig_top = fig_tops[rows].max()
        fig_left = fig_lefts[cols].min()
        fig_right = fig_rights[cols].max()
        return Bbox.from_extents(fig_left, fig_bottom, fig_right, fig_top)

    def get_topmost_subplotspec(self):
        """
        Return the topmost `SubplotSpec` instance associated with the subplot.
        """
        gridspec = self.get_gridspec()
        if hasattr(gridspec, 'get_topmost_subplotspec'):
            return gridspec.get_topmost_subplotspec()
        else:
            return self

    def __eq__(self, other):
        """
        Two SubplotSpecs are considered equal if they refer to the same
        position(s) in the same `GridSpec`.
        """
        return (self._gridspec, self.num1, self.num2) == (getattr(other, '_gridspec', object()), getattr(other, 'num1', object()), getattr(other, 'num2', object()))

    def __hash__(self):
        return hash((self._gridspec, self.num1, self.num2))

    def subgridspec(self, nrows, ncols, **kwargs):
        """
        Create a GridSpec within this subplot.

        The created `.GridSpecFromSubplotSpec` will have this `SubplotSpec` as
        a parent.

        Parameters
        ----------
        nrows : int
            Number of rows in grid.

        ncols : int
            Number of columns in grid.

        Returns
        -------
        `.GridSpecFromSubplotSpec`

        Other Parameters
        ----------------
        **kwargs
            All other parameters are passed to `.GridSpecFromSubplotSpec`.

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        Adding three subplots in the space occupied by a single subplot::

            fig = plt.figure()
            gs0 = fig.add_gridspec(3, 1)
            ax1 = fig.add_subplot(gs0[0])
            ax2 = fig.add_subplot(gs0[1])
            gssub = gs0[2].subgridspec(1, 3)
            for i in range(3):
                fig.add_subplot(gssub[0, i])
        """
        return GridSpecFromSubplotSpec(nrows, ncols, self, **kwargs)