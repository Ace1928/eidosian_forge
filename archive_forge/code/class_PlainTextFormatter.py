import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
class PlainTextFormatter(BaseFormatter):
    """The default pretty-printer.

    This uses :mod:`IPython.lib.pretty` to compute the format data of
    the object. If the object cannot be pretty printed, :func:`repr` is used.
    See the documentation of :mod:`IPython.lib.pretty` for details on
    how to write pretty printers.  Here is a simple example::

        def dtype_pprinter(obj, p, cycle):
            if cycle:
                return p.text('dtype(...)')
            if hasattr(obj, 'fields'):
                if obj.fields is None:
                    p.text(repr(obj))
                else:
                    p.begin_group(7, 'dtype([')
                    for i, field in enumerate(obj.descr):
                        if i > 0:
                            p.text(',')
                            p.breakable()
                        p.pretty(field)
                    p.end_group(7, '])')
    """
    format_type = Unicode('text/plain')
    enabled = Bool(True).tag(config=False)
    max_seq_length = Integer(pretty.MAX_SEQ_LENGTH, help='Truncate large collections (lists, dicts, tuples, sets) to this size.\n        \n        Set to 0 to disable truncation.\n        ').tag(config=True)
    print_method = ObjectName('_repr_pretty_')
    pprint = Bool(True).tag(config=True)
    verbose = Bool(False).tag(config=True)
    max_width = Integer(79).tag(config=True)
    newline = Unicode('\n').tag(config=True)
    float_format = Unicode('%r')
    float_precision = CUnicode('').tag(config=True)

    @observe('float_precision')
    def _float_precision_changed(self, change):
        """float_precision changed, set float_format accordingly.

        float_precision can be set by int or str.
        This will set float_format, after interpreting input.
        If numpy has been imported, numpy print precision will also be set.

        integer `n` sets format to '%.nf', otherwise, format set directly.

        An empty string returns to defaults (repr for float, 8 for numpy).

        This parameter can be set via the '%precision' magic.
        """
        new = change['new']
        if '%' in new:
            fmt = new
            try:
                fmt % 3.14159
            except Exception as e:
                raise ValueError('Precision must be int or format string, not %r' % new) from e
        elif new:
            try:
                i = int(new)
                assert i >= 0
            except ValueError as e:
                raise ValueError('Precision must be int or format string, not %r' % new) from e
            except AssertionError as e:
                raise ValueError('int precision must be non-negative, not %r' % i) from e
            fmt = '%%.%if' % i
            if 'numpy' in sys.modules:
                import numpy
                numpy.set_printoptions(precision=i)
        else:
            fmt = '%r'
            if 'numpy' in sys.modules:
                import numpy
                numpy.set_printoptions(precision=8)
        self.float_format = fmt

    @default('singleton_printers')
    def _singleton_printers_default(self):
        return pretty._singleton_pprinters.copy()

    @default('type_printers')
    def _type_printers_default(self):
        d = pretty._type_pprinters.copy()
        d[float] = lambda obj, p, cycle: p.text(self.float_format % obj)
        if 'numpy' in sys.modules:
            import numpy
            d[numpy.float64] = lambda obj, p, cycle: p.text(self.float_format % obj)
        return d

    @default('deferred_printers')
    def _deferred_printers_default(self):
        return pretty._deferred_type_pprinters.copy()

    @catch_format_error
    def __call__(self, obj):
        """Compute the pretty representation of the object."""
        if not self.pprint:
            return repr(obj)
        else:
            stream = StringIO()
            printer = pretty.RepresentationPrinter(stream, self.verbose, self.max_width, self.newline, max_seq_length=self.max_seq_length, singleton_pprinters=self.singleton_printers, type_pprinters=self.type_printers, deferred_pprinters=self.deferred_printers)
            printer.pretty(obj)
            printer.flush()
            return stream.getvalue()