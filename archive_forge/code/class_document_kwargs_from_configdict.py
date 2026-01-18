import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
class document_kwargs_from_configdict(object):
    """Decorator to append the documentation of a ConfigDict to the docstring

    This adds the documentation of the specified :py:class:`ConfigDict`
    (using the :py:class:`numpydoc_ConfigFormatter` formatter) to the
    decorated object's docstring.

    Parameters
    ----------
    config : ConfigDict or str
        the :py:class:`ConfigDict` to document.  If a ``str``, then the
        :py:class:`ConfigDict` is obtained by retrieving the named
        attribute from the decorated object (thereby enabling
        documenting class objects whose ``__init__`` keyword arguments
        are processed by a :py:class:`ConfigDict` class attribute)

    section : str
        the section header to preface config documentation with

    indent_spacing : int
        number of spaces to indent each block of documentation

    width : int
        total documentation width in characters (for wrapping paragraphs)

    doc : str, optional
        the initial docstring to append the ConfigDict documentation to.
        If None, then the decorated object's ``__doc__`` will be used.

    Examples
    --------

    >>> from pyomo.common.config import (
    ...     ConfigDict, ConfigValue, document_kwargs_from_configdict
    ... )
    >>> class MyClass(object):
    ...     CONFIG = ConfigDict()
    ...     CONFIG.declare('iterlim', ConfigValue(
    ...         default=3000,
    ...         domain=int,
    ...         doc="Iteration limit.  Specify None for no limit"
    ...     ))
    ...     CONFIG.declare('tee', ConfigValue(
    ...         domain=bool,
    ...         doc="If True, stream the solver output to the console"
    ...     ))
    ...
    ...     @document_kwargs_from_configdict(CONFIG)
    ...     def solve(self, **kwargs):
    ...         config = self.CONFIG(kwargs)
    ...         # ...
    ...
    >>> help(MyClass.solve)
    Help on function solve:
    <BLANKLINE>
    solve(self, **kwargs)
        Keyword Arguments
        -----------------
        iterlim: int, default=3000
            Iteration limit.  Specify None for no limit
    <BLANKLINE>
        tee: bool, optional
            If True, stream the solver output to the console

    """

    def __init__(self, config, section='Keyword Arguments', indent_spacing=4, width=78, visibility=None, doc=None):
        if '\n' not in section:
            section += '\n' + '-' * len(section) + '\n'
        self.config = config
        self.section = section
        self.indent_spacing = indent_spacing
        self.width = width
        self.visibility = visibility
        self.doc = doc

    def __call__(self, fcn):
        if isinstance(self.config, str):
            self.config = getattr(fcn, self.config)
        if self.doc is not None:
            doc = inspect.cleandoc(self.doc)
        elif fcn.__doc__:
            doc = inspect.cleandoc(fcn.__doc__)
        else:
            doc = ''
        if doc:
            if not doc.endswith('\n'):
                doc += '\n\n'
            else:
                doc += '\n'
        fcn.__doc__ = doc + f'{self.section}' + self.config.generate_documentation(indent_spacing=self.indent_spacing, width=self.width, visibility=self.visibility, format='numpydoc')
        return fcn