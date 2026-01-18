from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
class DesignMatrix(np.ndarray):
    """A simple numpy array subclass that carries design matrix metadata.

    .. attribute:: design_info

       A :class:`DesignInfo` object containing metadata about this design
       matrix.

    This class also defines a fancy __repr__ method with labeled
    columns. Otherwise it is identical to a regular numpy ndarray.

    .. warning::

       You should never check for this class using
       :func:`isinstance`. Limitations of the numpy API mean that it is
       impossible to prevent the creation of numpy arrays that have type
       DesignMatrix, but that are not actually design matrices (and such
       objects will behave like regular ndarrays in every way). Instead, check
       for the presence of a ``.design_info`` attribute -- this will be
       present only on "real" DesignMatrix objects.
    """

    def __new__(cls, input_array, design_info=None, default_column_prefix='column'):
        """Create a DesignMatrix, or cast an existing matrix to a DesignMatrix.

        A call like::

          DesignMatrix(my_array)

        will convert an arbitrary array_like object into a DesignMatrix.

        The return from this function is guaranteed to be a two-dimensional
        ndarray with a real-valued floating point dtype, and a
        ``.design_info`` attribute which matches its shape. If the
        `design_info` argument is not given, then one is created via
        :meth:`DesignInfo.from_array` using the given
        `default_column_prefix`.

        Depending on the input array, it is possible this will pass through
        its input unchanged, or create a view.
        """
        if isinstance(input_array, DesignMatrix) and hasattr(input_array, 'design_info'):
            return input_array
        self = atleast_2d_column_default(input_array).view(cls)
        if safe_issubdtype(self.dtype, np.integer):
            self = np.asarray(self, dtype=float).view(cls)
        if self.ndim > 2:
            raise ValueError('DesignMatrix must be 2d')
        assert self.ndim == 2
        if design_info is None:
            design_info = DesignInfo.from_array(self, default_column_prefix)
        if len(design_info.column_names) != self.shape[1]:
            raise ValueError('wrong number of column names for design matrix (got %s, wanted %s)' % (len(design_info.column_names), self.shape[1]))
        self.design_info = design_info
        if not safe_issubdtype(self.dtype, np.floating):
            raise ValueError('design matrix must be real-valued floating point')
        return self
    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        if not hasattr(self, 'design_info'):
            p.pretty(np.asarray(self))
            return
        assert not cycle
        MAX_TOTAL_WIDTH = 78
        SEP = 2
        INDENT = 2
        MAX_ROWS = 30
        PRECISION = 5
        names = self.design_info.column_names
        column_name_widths = [len(name) for name in names]
        min_total_width = INDENT + SEP * (self.shape[1] - 1) + np.sum(column_name_widths)
        if min_total_width <= MAX_TOTAL_WIDTH:
            printable_part = np.asarray(self)[:MAX_ROWS, :]
            formatted_cols = [_format_float_column(PRECISION, printable_part[:, i]) for i in range(self.shape[1])]

            def max_width(col):
                assert col.ndim == 1
                if not col.shape[0]:
                    return 0
                else:
                    return max([len(s) for s in col])
            column_num_widths = [max_width(col) for col in formatted_cols]
            column_widths = [max(name_width, num_width) for name_width, num_width in zip(column_name_widths, column_num_widths)]
            total_width = INDENT + SEP * (self.shape[1] - 1) + np.sum(column_widths)
            print_numbers = total_width < MAX_TOTAL_WIDTH
        else:
            print_numbers = False
        p.begin_group(INDENT, 'DesignMatrix with shape %s' % (self.shape,))
        p.breakable('\n' + ' ' * p.indentation)
        if print_numbers:
            sep = ' ' * SEP
            for row in [names] + list(zip(*formatted_cols)):
                cells = [cell.rjust(width) for width, cell in zip(column_widths, row)]
                p.text(sep.join(cells))
                p.text('\n' + ' ' * p.indentation)
            if MAX_ROWS < self.shape[0]:
                p.text('[%s rows omitted]' % (self.shape[0] - MAX_ROWS,))
                p.text('\n' + ' ' * p.indentation)
        else:
            p.begin_group(2, 'Columns:')
            p.breakable('\n' + ' ' * p.indentation)
            p.pretty(names)
            p.end_group(2, '')
            p.breakable('\n' + ' ' * p.indentation)
        p.begin_group(2, 'Terms:')
        p.breakable('\n' + ' ' * p.indentation)
        for term_name, span in six.iteritems(self.design_info.term_name_slices):
            if span.start != 0:
                p.breakable(', ')
            p.pretty(term_name)
            if span.stop - span.start == 1:
                coltext = 'column %s' % (span.start,)
            else:
                coltext = 'columns %s:%s' % (span.start, span.stop)
            p.text(' (%s)' % (coltext,))
        p.end_group(2, '')
        if not print_numbers or self.shape[0] > MAX_ROWS:
            p.breakable('\n' + ' ' * p.indentation)
            p.text('(to view full data, use np.asarray(this_obj))')
        p.end_group(INDENT, '')
    __reduce__ = no_pickling