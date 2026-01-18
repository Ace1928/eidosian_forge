from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level figure instance.
    """
    required_interactive_framework = None
    manager_class = _api.classproperty(lambda cls: FigureManagerBase)
    events = ['resize_event', 'draw_event', 'key_press_event', 'key_release_event', 'button_press_event', 'button_release_event', 'scroll_event', 'motion_notify_event', 'pick_event', 'figure_enter_event', 'figure_leave_event', 'axes_enter_event', 'axes_leave_event', 'close_event']
    fixed_dpi = None
    filetypes = _default_filetypes

    @_api.classproperty
    def supports_blit(cls):
        """If this Canvas sub-class supports blitting."""
        return hasattr(cls, 'copy_from_bbox') and hasattr(cls, 'restore_region')

    def __init__(self, figure=None):
        from matplotlib.figure import Figure
        self._fix_ipython_backend2gui()
        self._is_idle_drawing = True
        self._is_saving = False
        if figure is None:
            figure = Figure()
        figure.set_canvas(self)
        self.figure = figure
        self.manager = None
        self.widgetlock = widgets.LockDraw()
        self._button = None
        self._key = None
        self.mouse_grabber = None
        self.toolbar = None
        self._is_idle_drawing = False
        figure._original_dpi = figure.dpi
        self._device_pixel_ratio = 1
        super().__init__()
    callbacks = property(lambda self: self.figure._canvas_callbacks)
    button_pick_id = property(lambda self: self.figure._button_pick_id)
    scroll_pick_id = property(lambda self: self.figure._scroll_pick_id)

    @classmethod
    @functools.cache
    def _fix_ipython_backend2gui(cls):
        if sys.modules.get('IPython') is None:
            return
        import IPython
        ip = IPython.get_ipython()
        if not ip:
            return
        from IPython.core import pylabtools as pt
        if not hasattr(pt, 'backend2gui') or not hasattr(ip, 'enable_matplotlib'):
            return
        backend2gui_rif = {'qt': 'qt', 'gtk3': 'gtk3', 'gtk4': 'gtk4', 'wx': 'wx', 'macosx': 'osx'}.get(cls.required_interactive_framework)
        if backend2gui_rif:
            if _is_non_interactive_terminal_ipython(ip):
                ip.enable_gui(backend2gui_rif)

    @classmethod
    def new_manager(cls, figure, num):
        """
        Create a new figure manager for *figure*, using this canvas class.

        Notes
        -----
        This method should not be reimplemented in subclasses.  If
        custom manager creation logic is needed, please reimplement
        ``FigureManager.create_with_canvas``.
        """
        return cls.manager_class.create_with_canvas(cls, figure, num)

    @contextmanager
    def _idle_draw_cntx(self):
        self._is_idle_drawing = True
        try:
            yield
        finally:
            self._is_idle_drawing = False

    def is_saving(self):
        """
        Return whether the renderer is in the process of saving
        to a file, rather than rendering for an on-screen buffer.
        """
        return self._is_saving

    def blit(self, bbox=None):
        """Blit the canvas in bbox (default entire canvas)."""

    def inaxes(self, xy):
        """
        Return the topmost visible `~.axes.Axes` containing the point *xy*.

        Parameters
        ----------
        xy : (float, float)
            (x, y) pixel positions from left/bottom of the canvas.

        Returns
        -------
        `~matplotlib.axes.Axes` or None
            The topmost visible Axes containing the point, or None if there
            is no Axes at the point.
        """
        axes_list = [a for a in self.figure.get_axes() if a.patch.contains_point(xy) and a.get_visible()]
        if axes_list:
            axes = cbook._topmost_artist(axes_list)
        else:
            axes = None
        return axes

    def grab_mouse(self, ax):
        """
        Set the child `~.axes.Axes` which is grabbing the mouse events.

        Usually called by the widgets themselves. It is an error to call this
        if the mouse is already grabbed by another Axes.
        """
        if self.mouse_grabber not in (None, ax):
            raise RuntimeError('Another Axes already grabs mouse input')
        self.mouse_grabber = ax

    def release_mouse(self, ax):
        """
        Release the mouse grab held by the `~.axes.Axes` *ax*.

        Usually called by the widgets. It is ok to call this even if *ax*
        doesn't have the mouse grab currently.
        """
        if self.mouse_grabber is ax:
            self.mouse_grabber = None

    def set_cursor(self, cursor):
        """
        Set the current cursor.

        This may have no effect if the backend does not display anything.

        If required by the backend, this method should trigger an update in
        the backend event loop after the cursor is set, as this method may be
        called e.g. before a long-running task during which the GUI is not
        updated.

        Parameters
        ----------
        cursor : `.Cursors`
            The cursor to display over the canvas. Note: some backends may
            change the cursor for the entire window.
        """

    def draw(self, *args, **kwargs):
        """
        Render the `.Figure`.

        This method must walk the artist tree, even if no output is produced,
        because it triggers deferred work that users may want to access
        before saving output to disk. For example computing limits,
        auto-limits, and tick values.
        """

    def draw_idle(self, *args, **kwargs):
        """
        Request a widget redraw once control returns to the GUI event loop.

        Even if multiple calls to `draw_idle` occur before control returns
        to the GUI event loop, the figure will only be rendered once.

        Notes
        -----
        Backends may choose to override the method and implement their own
        strategy to prevent multiple renderings.

        """
        if not self._is_idle_drawing:
            with self._idle_draw_cntx():
                self.draw(*args, **kwargs)

    @property
    def device_pixel_ratio(self):
        """
        The ratio of physical to logical pixels used for the canvas on screen.

        By default, this is 1, meaning physical and logical pixels are the same
        size. Subclasses that support High DPI screens may set this property to
        indicate that said ratio is different. All Matplotlib interaction,
        unless working directly with the canvas, remains in logical pixels.

        """
        return self._device_pixel_ratio

    def _set_device_pixel_ratio(self, ratio):
        """
        Set the ratio of physical to logical pixels used for the canvas.

        Subclasses that support High DPI screens can set this property to
        indicate that said ratio is different. The canvas itself will be
        created at the physical size, while the client side will use the
        logical size. Thus the DPI of the Figure will change to be scaled by
        this ratio. Implementations that support High DPI screens should use
        physical pixels for events so that transforms back to Axes space are
        correct.

        By default, this is 1, meaning physical and logical pixels are the same
        size.

        Parameters
        ----------
        ratio : float
            The ratio of logical to physical pixels used for the canvas.

        Returns
        -------
        bool
            Whether the ratio has changed. Backends may interpret this as a
            signal to resize the window, repaint the canvas, or change any
            other relevant properties.
        """
        if self._device_pixel_ratio == ratio:
            return False
        dpi = ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)
        self._device_pixel_ratio = ratio
        return True

    def get_width_height(self, *, physical=False):
        """
        Return the figure width and height in integral points or pixels.

        When the figure is used on High DPI screens (and the backend supports
        it), the truncation to integers occurs after scaling by the device
        pixel ratio.

        Parameters
        ----------
        physical : bool, default: False
            Whether to return true physical pixels or logical pixels. Physical
            pixels may be used by backends that support HiDPI, but still
            configure the canvas using its actual size.

        Returns
        -------
        width, height : int
            The size of the figure, in points or pixels, depending on the
            backend.
        """
        return tuple((int(size / (1 if physical else self.device_pixel_ratio)) for size in self.figure.bbox.max))

    @classmethod
    def get_supported_filetypes(cls):
        """Return dict of savefig file formats supported by this backend."""
        return cls.filetypes

    @classmethod
    def get_supported_filetypes_grouped(cls):
        """
        Return a dict of savefig file formats supported by this backend,
        where the keys are a file type name, such as 'Joint Photographic
        Experts Group', and the values are a list of filename extensions used
        for that filetype, such as ['jpg', 'jpeg'].
        """
        groupings = {}
        for ext, name in cls.filetypes.items():
            groupings.setdefault(name, []).append(ext)
            groupings[name].sort()
        return groupings

    @contextmanager
    def _switch_canvas_and_return_print_method(self, fmt, backend=None):
        """
        Context manager temporarily setting the canvas for saving the figure::

            with canvas._switch_canvas_and_return_print_method(fmt, backend) \\
                    as print_method:
                # ``print_method`` is a suitable ``print_{fmt}`` method, and
                # the figure's canvas is temporarily switched to the method's
                # canvas within the with... block.  ``print_method`` is also
                # wrapped to suppress extra kwargs passed by ``print_figure``.

        Parameters
        ----------
        fmt : str
            If *backend* is None, then determine a suitable canvas class for
            saving to format *fmt* -- either the current canvas class, if it
            supports *fmt*, or whatever `get_registered_canvas_class` returns;
            switch the figure canvas to that canvas class.
        backend : str or None, default: None
            If not None, switch the figure canvas to the ``FigureCanvas`` class
            of the given backend.
        """
        canvas = None
        if backend is not None:
            canvas_class = importlib.import_module(cbook._backend_module_name(backend)).FigureCanvas
            if not hasattr(canvas_class, f'print_{fmt}'):
                raise ValueError(f'The {backend!r} backend does not support {fmt} output')
            canvas = canvas_class(self.figure)
        elif hasattr(self, f'print_{fmt}'):
            canvas = self
        else:
            canvas_class = get_registered_canvas_class(fmt)
            if canvas_class is None:
                raise ValueError('Format {!r} is not supported (supported formats: {})'.format(fmt, ', '.join(sorted(self.get_supported_filetypes()))))
            canvas = canvas_class(self.figure)
        canvas._is_saving = self._is_saving
        meth = getattr(canvas, f'print_{fmt}')
        mod = meth.func.__module__ if hasattr(meth, 'func') else meth.__module__
        if mod.startswith(('matplotlib.', 'mpl_toolkits.')):
            optional_kws = {'dpi', 'facecolor', 'edgecolor', 'orientation', 'bbox_inches_restore'}
            skip = optional_kws - {*inspect.signature(meth).parameters}
            print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(*args, **{k: v for k, v in kwargs.items() if k not in skip}))
        else:
            print_method = meth
        try:
            yield print_method
        finally:
            self.figure.canvas = self

    def print_figure(self, filename, dpi=None, facecolor=None, edgecolor=None, orientation='portrait', format=None, *, bbox_inches=None, pad_inches=None, bbox_extra_artists=None, backend=None, **kwargs):
        """
        Render the figure to hardcopy. Set the figure patch face and edge
        colors.  This is useful because some of the GUIs have a gray figure
        face color background and you'll probably want to override this on
        hardcopy.

        Parameters
        ----------
        filename : str or path-like or file-like
            The file where the figure is saved.

        dpi : float, default: :rc:`savefig.dpi`
            The dots per inch to save the figure in.

        facecolor : color or 'auto', default: :rc:`savefig.facecolor`
            The facecolor of the figure.  If 'auto', use the current figure
            facecolor.

        edgecolor : color or 'auto', default: :rc:`savefig.edgecolor`
            The edgecolor of the figure.  If 'auto', use the current figure
            edgecolor.

        orientation : {'landscape', 'portrait'}, default: 'portrait'
            Only currently applies to PostScript printing.

        format : str, optional
            Force a specific file format. If not given, the format is inferred
            from the *filename* extension, and if that fails from
            :rc:`savefig.format`.

        bbox_inches : 'tight' or `.Bbox`, default: :rc:`savefig.bbox`
            Bounding box in inches: only the given portion of the figure is
            saved.  If 'tight', try to figure out the tight bbox of the figure.

        pad_inches : float or 'layout', default: :rc:`savefig.pad_inches`
            Amount of padding in inches around the figure when bbox_inches is
            'tight'. If 'layout' use the padding from the constrained or
            compressed layout engine; ignored if one of those engines is not in
            use.

        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
            A list of extra artists that will be considered when the
            tight bbox is calculated.

        backend : str, optional
            Use a non-default backend to render the file, e.g. to render a
            png file with the "cairo" backend rather than the default "agg",
            or a pdf file with the "pgf" backend rather than the default
            "pdf".  Note that the default backend is normally sufficient.  See
            :ref:`the-builtin-backends` for a list of valid backends for each
            file format.  Custom backends can be referenced as "module://...".
        """
        if format is None:
            if isinstance(filename, os.PathLike):
                filename = os.fspath(filename)
            if isinstance(filename, str):
                format = os.path.splitext(filename)[1][1:]
            if format is None or format == '':
                format = self.get_default_filetype()
                if isinstance(filename, str):
                    filename = filename.rstrip('.') + '.' + format
        format = format.lower()
        if dpi is None:
            dpi = rcParams['savefig.dpi']
        if dpi == 'figure':
            dpi = getattr(self.figure, '_original_dpi', self.figure.dpi)
        if kwargs.get('papertype') == 'auto':
            _api.warn_deprecated('3.8', name="papertype='auto'", addendum="Pass an explicit paper type, 'figure', or omit the *papertype* argument entirely.")
        with cbook._setattr_cm(self, manager=None), self._switch_canvas_and_return_print_method(format, backend) as print_method, cbook._setattr_cm(self.figure, dpi=dpi), cbook._setattr_cm(self.figure.canvas, _device_pixel_ratio=1), cbook._setattr_cm(self.figure.canvas, _is_saving=True), ExitStack() as stack:
            for prop in ['facecolor', 'edgecolor']:
                color = locals()[prop]
                if color is None:
                    color = rcParams[f'savefig.{prop}']
                if not cbook._str_equal(color, 'auto'):
                    stack.enter_context(self.figure._cm_set(**{prop: color}))
            if bbox_inches is None:
                bbox_inches = rcParams['savefig.bbox']
            layout_engine = self.figure.get_layout_engine()
            if layout_engine is not None or bbox_inches == 'tight':
                renderer = _get_renderer(self.figure, functools.partial(print_method, orientation=orientation))
                with getattr(renderer, '_draw_disabled', nullcontext)():
                    self.figure.draw(renderer)
            if bbox_inches:
                if bbox_inches == 'tight':
                    bbox_inches = self.figure.get_tightbbox(renderer, bbox_extra_artists=bbox_extra_artists)
                    if isinstance(layout_engine, ConstrainedLayoutEngine) and pad_inches == 'layout':
                        h_pad = layout_engine.get()['h_pad']
                        w_pad = layout_engine.get()['w_pad']
                    else:
                        if pad_inches in [None, 'layout']:
                            pad_inches = rcParams['savefig.pad_inches']
                        h_pad = w_pad = pad_inches
                    bbox_inches = bbox_inches.padded(w_pad, h_pad)
                restore_bbox = _tight_bbox.adjust_bbox(self.figure, bbox_inches, self.figure.canvas.fixed_dpi)
                _bbox_inches_restore = (bbox_inches, restore_bbox)
            else:
                _bbox_inches_restore = None
            stack.enter_context(self.figure._cm_set(layout_engine='none'))
            try:
                with cbook._setattr_cm(self.figure, dpi=dpi):
                    result = print_method(filename, facecolor=facecolor, edgecolor=edgecolor, orientation=orientation, bbox_inches_restore=_bbox_inches_restore, **kwargs)
            finally:
                if bbox_inches and restore_bbox:
                    restore_bbox()
            return result

    @classmethod
    def get_default_filetype(cls):
        """
        Return the default savefig file format as specified in
        :rc:`savefig.format`.

        The returned string does not include a period. This method is
        overridden in backends that only support a single file type.
        """
        return rcParams['savefig.format']

    def get_default_filename(self):
        """
        Return a string, which includes extension, suitable for use as
        a default filename.
        """
        basename = self.manager.get_window_title() if self.manager is not None else ''
        basename = (basename or 'image').replace(' ', '_')
        filetype = self.get_default_filetype()
        filename = basename + '.' + filetype
        return filename

    @_api.deprecated('3.8')
    def switch_backends(self, FigureCanvasClass):
        """
        Instantiate an instance of FigureCanvasClass

        This is used for backend switching, e.g., to instantiate a
        FigureCanvasPS from a FigureCanvasGTK.  Note, deep copying is
        not done, so any changes to one of the instances (e.g., setting
        figure size or line props), will be reflected in the other
        """
        newCanvas = FigureCanvasClass(self.figure)
        newCanvas._is_saving = self._is_saving
        return newCanvas

    def mpl_connect(self, s, func):
        """
        Bind function *func* to event *s*.

        Parameters
        ----------
        s : str
            One of the following events ids:

            - 'button_press_event'
            - 'button_release_event'
            - 'draw_event'
            - 'key_press_event'
            - 'key_release_event'
            - 'motion_notify_event'
            - 'pick_event'
            - 'resize_event'
            - 'scroll_event'
            - 'figure_enter_event',
            - 'figure_leave_event',
            - 'axes_enter_event',
            - 'axes_leave_event'
            - 'close_event'.

        func : callable
            The callback function to be executed, which must have the
            signature::

                def func(event: Event) -> Any

            For the location events (button and key press/release), if the
            mouse is over the Axes, the ``inaxes`` attribute of the event will
            be set to the `~matplotlib.axes.Axes` the event occurs is over, and
            additionally, the variables ``xdata`` and ``ydata`` attributes will
            be set to the mouse location in data coordinates.  See `.KeyEvent`
            and `.MouseEvent` for more info.

            .. note::

                If func is a method, this only stores a weak reference to the
                method. Thus, the figure does not influence the lifetime of
                the associated object. Usually, you want to make sure that the
                object is kept alive throughout the lifetime of the figure by
                holding a reference to it.

        Returns
        -------
        cid
            A connection id that can be used with
            `.FigureCanvasBase.mpl_disconnect`.

        Examples
        --------
        ::

            def on_press(event):
                print('you pressed', event.button, event.xdata, event.ydata)

            cid = canvas.mpl_connect('button_press_event', on_press)
        """
        return self.callbacks.connect(s, func)

    def mpl_disconnect(self, cid):
        """
        Disconnect the callback with id *cid*.

        Examples
        --------
        ::

            cid = canvas.mpl_connect('button_press_event', on_press)
            # ... later
            canvas.mpl_disconnect(cid)
        """
        self.callbacks.disconnect(cid)
    _timer_cls = TimerBase

    def new_timer(self, interval=None, callbacks=None):
        """
        Create a new backend-specific subclass of `.Timer`.

        This is useful for getting periodic events through the backend's native
        event loop.  Implemented only for backends with GUIs.

        Parameters
        ----------
        interval : int
            Timer interval in milliseconds.

        callbacks : list[tuple[callable, tuple, dict]]
            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
            will be executed by the timer every *interval*.

            Callbacks which return ``False`` or ``0`` will be removed from the
            timer.

        Examples
        --------
        >>> timer = fig.canvas.new_timer(callbacks=[(f1, (1,), {'a': 3})])
        """
        return self._timer_cls(interval=interval, callbacks=callbacks)

    def flush_events(self):
        """
        Flush the GUI events for the figure.

        Interactive backends need to reimplement this method.
        """

    def start_event_loop(self, timeout=0):
        """
        Start a blocking event loop.

        Such an event loop is used by interactive functions, such as
        `~.Figure.ginput` and `~.Figure.waitforbuttonpress`, to wait for
        events.

        The event loop blocks until a callback function triggers
        `stop_event_loop`, or *timeout* is reached.

        If *timeout* is 0 or negative, never timeout.

        Only interactive backends need to reimplement this method and it relies
        on `flush_events` being properly implemented.

        Interactive backends should implement this in a more native way.
        """
        if timeout <= 0:
            timeout = np.inf
        timestep = 0.01
        counter = 0
        self._looping = True
        while self._looping and counter * timestep < timeout:
            self.flush_events()
            time.sleep(timestep)
            counter += 1

    def stop_event_loop(self):
        """
        Stop the current blocking event loop.

        Interactive backends need to reimplement this to match
        `start_event_loop`
        """
        self._looping = False