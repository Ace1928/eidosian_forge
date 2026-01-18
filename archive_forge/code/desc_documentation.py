from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
Construct a :class:`ModelDesc` from a formula string.

        :arg tree_or_string: A formula string. (Or an unevaluated formula
          parse tree, but the API for generating those isn't public yet. Shh,
          it can be our secret.)
        :returns: A new :class:`ModelDesc`.
        