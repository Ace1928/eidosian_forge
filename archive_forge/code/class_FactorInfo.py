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
class FactorInfo(object):
    """A FactorInfo object is a simple class that provides some metadata about
    the role of a factor within a model. :attr:`DesignInfo.factor_infos` is
    a dictionary which maps factor objects to FactorInfo objects for each
    factor in the model.

    .. versionadded:: 0.4.0

    Attributes:

    .. attribute:: factor

       The factor object being described.

    .. attribute:: type

       The type of the factor -- either the string ``"numerical"`` or the
       string ``"categorical"``.

    .. attribute:: state

       An opaque object which holds the state needed to evaluate this
       factor on new data (e.g., for prediction). See
       :meth:`factor_protocol.eval`.

    .. attribute:: num_columns

       For numerical factors, the number of columns this factor produces. For
       categorical factors, this attribute will always be ``None``.

    .. attribute:: categories

       For categorical factors, a tuple of the possible categories this factor
       takes on, in order. For numerical factors, this attribute will always be
       ``None``.
    """

    def __init__(self, factor, type, state, num_columns=None, categories=None):
        self.factor = factor
        self.type = type
        if self.type not in ['numerical', 'categorical']:
            raise ValueError("FactorInfo.type must be 'numerical' or 'categorical', not %r" % (self.type,))
        self.state = state
        if self.type == 'numerical':
            if not isinstance(num_columns, six.integer_types):
                raise ValueError('For numerical factors, num_columns must be an integer')
            if categories is not None:
                raise ValueError('For numerical factors, categories must be None')
        else:
            assert self.type == 'categorical'
            if num_columns is not None:
                raise ValueError('For categorical factors, num_columns must be None')
            categories = tuple(categories)
        self.num_columns = num_columns
        self.categories = categories
    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        assert not cycle

        class FactorState(object):

            def __repr__(self):
                return '<factor state>'
        kwlist = [('factor', self.factor), ('type', self.type), ('state', FactorState())]
        if self.type == 'numerical':
            kwlist.append(('num_columns', self.num_columns))
        else:
            kwlist.append(('categories', self.categories))
        repr_pretty_impl(p, self, [], kwlist)
    __getstate__ = no_pickling