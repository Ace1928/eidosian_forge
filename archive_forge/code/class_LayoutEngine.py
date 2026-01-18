from contextlib import nullcontext
import matplotlib as mpl
from matplotlib._constrained_layout import do_constrained_layout
from matplotlib._tight_layout import (get_subplotspec_list,
class LayoutEngine:
    """
    Base class for Matplotlib layout engines.

    A layout engine can be passed to a figure at instantiation or at any time
    with `~.figure.Figure.set_layout_engine`.  Once attached to a figure, the
    layout engine ``execute`` function is called at draw time by
    `~.figure.Figure.draw`, providing a special draw-time hook.

    .. note::

       However, note that layout engines affect the creation of colorbars, so
       `~.figure.Figure.set_layout_engine` should be called before any
       colorbars are created.

    Currently, there are two properties of `LayoutEngine` classes that are
    consulted while manipulating the figure:

    - ``engine.colorbar_gridspec`` tells `.Figure.colorbar` whether to make the
       axes using the gridspec method (see `.colorbar.make_axes_gridspec`) or
       not (see `.colorbar.make_axes`);
    - ``engine.adjust_compatible`` stops `.Figure.subplots_adjust` from being
        run if it is not compatible with the layout engine.

    To implement a custom `LayoutEngine`:

    1. override ``_adjust_compatible`` and ``_colorbar_gridspec``
    2. override `LayoutEngine.set` to update *self._params*
    3. override `LayoutEngine.execute` with your implementation

    """
    _adjust_compatible = None
    _colorbar_gridspec = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._params = {}

    def set(self, **kwargs):
        """
        Set the parameters for the layout engine.
        """
        raise NotImplementedError

    @property
    def colorbar_gridspec(self):
        """
        Return a boolean if the layout engine creates colorbars using a
        gridspec.
        """
        if self._colorbar_gridspec is None:
            raise NotImplementedError
        return self._colorbar_gridspec

    @property
    def adjust_compatible(self):
        """
        Return a boolean if the layout engine is compatible with
        `~.Figure.subplots_adjust`.
        """
        if self._adjust_compatible is None:
            raise NotImplementedError
        return self._adjust_compatible

    def get(self):
        """
        Return copy of the parameters for the layout engine.
        """
        return dict(self._params)

    def execute(self, fig):
        """
        Execute the layout on the figure given by *fig*.
        """
        raise NotImplementedError