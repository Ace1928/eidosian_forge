from collections.abc import Iterable, Mapping
from inspect import signature, Parameter
from inspect import getcallargs
from inspect import getfullargspec as check_argspec
import sys
from IPython import get_ipython
from . import (Widget, ValueWidget, Text,
from IPython.display import display, clear_output
from traitlets import HasTraits, Any, Unicode, observe
from numbers import Real, Integral
from warnings import warn
class _InteractFactory:
    """
    Factory for instances of :class:`interactive`.

    This class is needed to support options like::

        >>> @interact.options(manual=True)
        ... def greeting(text="World"):
        ...     print("Hello {}".format(text))

    Parameters
    ----------
    cls : class
        The subclass of :class:`interactive` to construct.
    options : dict
        A dict of options used to construct the interactive
        function. By default, this is returned by
        ``cls.default_options()``.
    kwargs : dict
        A dict of **kwargs to use for widgets.
    """

    def __init__(self, cls, options, kwargs={}):
        self.cls = cls
        self.opts = options
        self.kwargs = kwargs

    def widget(self, f):
        """
        Return an interactive function widget for the given function.

        The widget is only constructed, not displayed nor attached to
        the function.

        Returns
        -------
        An instance of ``self.cls`` (typically :class:`interactive`).

        Parameters
        ----------
        f : function
            The function to which the interactive widgets are tied.
        """
        return self.cls(f, self.opts, **self.kwargs)

    def __call__(self, __interact_f=None, **kwargs):
        """
        Make the given function interactive by adding and displaying
        the corresponding :class:`interactive` widget.

        Expects the first argument to be a function. Parameters to this
        function are widget abbreviations passed in as keyword arguments
        (``**kwargs``). Can be used as a decorator (see examples).

        Returns
        -------
        f : __interact_f with interactive widget attached to it.

        Parameters
        ----------
        __interact_f : function
            The function to which the interactive widgets are tied. The `**kwargs`
            should match the function signature. Passed to :func:`interactive()`
        **kwargs : various, optional
            An interactive widget is created for each keyword argument that is a
            valid widget abbreviation. Passed to :func:`interactive()`

        Examples
        --------
        Render an interactive text field that shows the greeting with the passed in
        text::

            # 1. Using interact as a function
            def greeting(text="World"):
                print("Hello {}".format(text))
            interact(greeting, text="Jupyter Widgets")

            # 2. Using interact as a decorator
            @interact
            def greeting(text="World"):
                print("Hello {}".format(text))

            # 3. Using interact as a decorator with named parameters
            @interact(text="Jupyter Widgets")
            def greeting(text="World"):
                print("Hello {}".format(text))

        Render an interactive slider widget and prints square of number::

            # 1. Using interact as a function
            def square(num=1):
                print("{} squared is {}".format(num, num*num))
            interact(square, num=5)

            # 2. Using interact as a decorator
            @interact
            def square(num=2):
                print("{} squared is {}".format(num, num*num))

            # 3. Using interact as a decorator with named parameters
            @interact(num=5)
            def square(num=2):
                print("{} squared is {}".format(num, num*num))
        """
        if kwargs:
            kw = dict(self.kwargs)
            kw.update(kwargs)
            self = type(self)(self.cls, self.opts, kw)
        f = __interact_f
        if f is None:
            return self
        w = self.widget(f)
        try:
            f.widget = w
        except AttributeError:
            f = lambda *args, **kwargs: __interact_f(*args, **kwargs)
            f.widget = w
        show_inline_matplotlib_plots()
        display(w)
        return f

    def options(self, **kwds):
        """
        Change options for interactive functions.

        Returns
        -------
        A new :class:`_InteractFactory` which will apply the
        options when called.
        """
        opts = dict(self.opts)
        for k in kwds:
            try:
                _ = opts[k]
            except KeyError:
                raise ValueError('invalid option {!r}'.format(k))
            opts[k] = kwds[k]
        return type(self)(self.cls, opts, self.kwargs)