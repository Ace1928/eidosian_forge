import logging
import math
import operator
import sys
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from pyomo.core.expr.visitor import clone_expression
from pyomo.environ import ConcreteModel, Param, Var, BooleanVar
from pyomo.gdp import Disjunct
def _run_iadd_cases(self, tests, op):
    self.assertEqual(len(tests), self.NUM_TESTS)
    ref = tests[0][:-2]
    for test_num, test in enumerate(tests):
        self.assertIs(test[-2], self.TEMPLATE[test_num])
        for i, j in zip(test[:-2], ref):
            self.assertIs(i, j)
    try:
        for test_num, test in enumerate(tests):
            ans = None
            args = test[:-1]
            result = test[-1]
            if result is self.SKIP:
                continue
            orig_args = list(args)
            orig_args_clone = [clone_expression(arg) for arg in args]
            try:
                mutable = [isinstance(arg, _MutableSumExpression) for arg in args]
                classes = [arg.__class__ for arg in args]
                with LoggingIntercept() as LOG:
                    ans = op(*args)
                if not any((arg is self.asbinary for arg in args)):
                    self.assertEqual(LOG.getvalue(), '')
                assertExpressionsEqual(self, result, ans)
                for i, arg in enumerate(args):
                    self.assertIs(arg, orig_args[i])
                    if mutable[i]:
                        if i:
                            self.assertFalse(isinstance(arg, _MutableSumExpression))
                            self.assertIsNot(arg.__class__, classes[i])
                            assertExpressionsEqual(self, _MutableSumExpression(arg.args), _MutableSumExpression(orig_args_clone[i].args))
                        else:
                            self.assertIsInstance(arg, _MutableSumExpression)
                            self.assertIs(arg, ans)
                    else:
                        self.assertIs(arg.__class__, classes[i])
                        assertExpressionsEqual(self, arg, orig_args_clone[i])
            except TypeError:
                if result is not NotImplemented:
                    raise
            except ZeroDivisionError:
                if result is not ZeroDivisionError:
                    raise
            except ValueError:
                if result is not ValueError:
                    raise
            else:
                for i, arg in enumerate(args):
                    if mutable[i]:
                        arg.__class__ = classes[i]
                        arg._args_ = orig_args_clone[i]._args_
                        arg._nargs = orig_args_clone[i]._nargs
    except:
        self._print_error(test_num, orig_args_clone + [result], ans)
        for i, arg in enumerate(args):
            if mutable[i]:
                arg.__class__ = classes[i]
                arg._args_ = orig_args_clone[i]._args_
                arg._nargs = orig_args_clone[i]._nargs
        raise