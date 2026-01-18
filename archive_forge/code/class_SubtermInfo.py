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
class SubtermInfo(object):
    """A SubtermInfo object is a simple metadata container describing a single
    primitive interaction and how it is coded in our design matrix. Our final
    design matrix is produced by coding each primitive interaction in order
    from left to right, and then stacking the resulting columns. For each
    :class:`Term`, we have one or more of these objects which describe how
    that term is encoded. :attr:`DesignInfo.term_codings` is a dictionary
    which maps term objects to lists of SubtermInfo objects.

    To code a primitive interaction, the following steps are performed:

    * Evaluate each factor on the provided data.
    * Encode each factor into one or more proto-columns. For numerical
      factors, these proto-columns are identical to whatever the factor
      evaluates to; for categorical factors, they are encoded using a
      specified contrast matrix.
    * Form all pairwise, elementwise products between proto-columns generated
      by different factors. (For example, if factor 1 generated proto-columns
      A and B, and factor 2 generated proto-columns C and D, then our final
      columns are ``A * C``, ``B * C``, ``A * D``, ``B * D``.)
    * The resulting columns are stored directly into the final design matrix.

    Sometimes multiple primitive interactions are needed to encode a single
    term; this occurs, for example, in the formula ``"1 + a:b"`` when ``a``
    and ``b`` are categorical. See :ref:`formulas-building` for full details.

    .. versionadded:: 0.4.0

    Attributes:

    .. attribute:: factors

       The factors which appear in this subterm's interaction.

    .. attribute:: contrast_matrices

       A dict mapping factor objects to :class:`ContrastMatrix` objects,
       describing how each categorical factor in this interaction is coded.

    .. attribute:: num_columns

       The number of design matrix columns which this interaction generates.

    """

    def __init__(self, factors, contrast_matrices, num_columns):
        self.factors = tuple(factors)
        factor_set = frozenset(factors)
        if not isinstance(contrast_matrices, dict):
            raise ValueError('contrast_matrices must be dict')
        for factor, contrast_matrix in six.iteritems(contrast_matrices):
            if factor not in factor_set:
                raise ValueError('Unexpected factor in contrast_matrices dict')
            if not isinstance(contrast_matrix, ContrastMatrix):
                raise ValueError('Expected a ContrastMatrix, not %r' % (contrast_matrix,))
        self.contrast_matrices = contrast_matrices
        if not isinstance(num_columns, six.integer_types):
            raise ValueError('num_columns must be an integer')
        self.num_columns = num_columns
    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        assert not cycle
        repr_pretty_impl(p, self, [], [('factors', self.factors), ('contrast_matrices', self.contrast_matrices), ('num_columns', self.num_columns)])
    __getstate__ = no_pickling