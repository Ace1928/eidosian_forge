import itertools
import numpy as np
from patsy import PatsyError
from patsy.categorical import C
from patsy.util import no_pickling, assert_no_pickling
class LookupFactor(object):
    """A simple factor class that simply looks up a named entry in the given
    data.

    Useful for programatically constructing formulas, and as a simple example
    of the factor protocol.  For details see
    :ref:`expert-model-specification`.

    Example::

      dmatrix(ModelDesc([], [Term([LookupFactor("x")])]), {"x": [1, 2, 3]})

    :arg varname: The name of this variable; used as a lookup key in the
      passed in data dictionary/DataFrame/whatever.
    :arg force_categorical: If True, then treat this factor as
      categorical. (Equivalent to using :func:`C` in a regular formula, but
      of course you can't do that with a :class:`LookupFactor`.
    :arg contrast: If given, the contrast to use; see :func:`C`. (Requires
      ``force_categorical=True``.)
    :arg levels: If given, the categorical levels; see :func:`C`. (Requires
      ``force_categorical=True``.)
    :arg origin: Either ``None``, or the :class:`Origin` of this factor for use
      in error reporting.

    .. versionadded:: 0.2.0
       The ``force_categorical`` and related arguments.
    """

    def __init__(self, varname, force_categorical=False, contrast=None, levels=None, origin=None):
        self._varname = varname
        self._force_categorical = force_categorical
        self._contrast = contrast
        self._levels = levels
        self.origin = origin
        if not self._force_categorical:
            if contrast is not None:
                raise ValueError('contrast= requires force_categorical=True')
            if levels is not None:
                raise ValueError('levels= requires force_categorical=True')

    def name(self):
        return self._varname

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._varname)

    def __eq__(self, other):
        return isinstance(other, LookupFactor) and self._varname == other._varname and (self._force_categorical == other._force_categorical) and (self._contrast == other._contrast) and (self._levels == other._levels)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((LookupFactor, self._varname, self._force_categorical, self._contrast, self._levels))

    def memorize_passes_needed(self, state, eval_env):
        return 0

    def memorize_chunk(self, state, which_pass, data):
        assert False

    def memorize_finish(self, state, which_pass):
        assert False

    def eval(self, memorize_state, data):
        value = data[self._varname]
        if self._force_categorical:
            value = C(value, contrast=self._contrast, levels=self._levels)
        return value
    __getstate__ = no_pickling