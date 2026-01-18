import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.block import IBlock, block, block_dict, block_list
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.piecewise_library.transforms import (
import pyomo.core.kernel.piecewise_library.transforms as transforms
from pyomo.core.kernel.piecewise_library.transforms_nd import (
import pyomo.core.kernel.piecewise_library.transforms_nd as transforms_nd
import pyomo.core.kernel.piecewise_library.util as util
class Test_piecewise(unittest.TestCase):

    def test_pickle(self):
        for key in transforms.registered_transforms:
            v = variable(lb=1, ub=3)
            p = transforms.piecewise([1, 2, 3], [1, 2, 1], input=v, validate=False, repn=key)
            self.assertEqual(p.parent, None)
            self.assertEqual(p.input.expr.parent, None)
            self.assertIs(p.input.expr, v)
            pup = pickle.loads(pickle.dumps(p))
            self.assertEqual(pup.parent, None)
            self.assertEqual(pup.input.expr.parent, None)
            self.assertIsNot(pup.input.expr, v)
            b = block()
            b.v = v
            b.p = p
            self.assertIs(p.parent, b)
            self.assertEqual(p.input.expr.parent, b)
            bup = pickle.loads(pickle.dumps(b))
            pup = bup.p
            self.assertIs(pup.parent, bup)
            self.assertEqual(pup.input.expr.parent, bup)
            self.assertIs(pup.input.expr, bup.v)
            self.assertIsNot(pup.input.expr, b.v)

    def test_call(self):
        g = PiecewiseLinearFunction([1], [0])
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 0)
        self.assertIs(type(f(1)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(1.1)
        g = PiecewiseLinearFunction([1, 2], [0, 4])
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 0)
        self.assertIs(type(f(1)), float)
        self.assertEqual(f(1.5), 2)
        self.assertIs(type(f(1.5)), float)
        self.assertEqual(f(2), 4)
        self.assertIs(type(f(2)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(2.1)
        g = PiecewiseLinearFunction([1, 1], [0, 1])
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 0)
        self.assertIs(type(f(1)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(1.1)
        g = PiecewiseLinearFunction([1, 2, 3], [1, 2, 1])
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertIs(type(f(1)), float)
        self.assertEqual(f(1.5), 1.5)
        self.assertIs(type(f(1.5)), float)
        self.assertEqual(f(2), 2)
        self.assertIs(type(f(2)), float)
        self.assertEqual(f(2.5), 1.5)
        self.assertIs(type(f(2.5)), float)
        self.assertEqual(f(3), 1)
        self.assertIs(type(f(3)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)
        g = PiecewiseLinearFunction([1, 2, 2, 3], [1, 2, 3, 4])
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertIs(type(f(1)), float)
        self.assertEqual(f(1.5), 1.5)
        self.assertIs(type(f(1.5)), float)
        self.assertEqual(f(2), 2)
        self.assertIs(type(f(2)), float)
        self.assertEqual(f(2.5), 3.5)
        self.assertIs(type(f(2.5)), float)
        self.assertEqual(f(3), 4)
        self.assertIs(type(f(3)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)
        g = PiecewiseLinearFunction([1, 1, 2, 3], [1, 2, 3, 4], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 2.5)
        self.assertEqual(f(2), 3)
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)
        g = PiecewiseLinearFunction([1, 2, 3, 3], [1, 2, 3, 4], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)
        g = PiecewiseLinearFunction([pmo.parameter(1), pmo.parameter(1), pmo.parameter(2), pmo.parameter(3)], [pmo.parameter(1), pmo.parameter(2), pmo.parameter(3), pmo.parameter(4)], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 2.5)
        self.assertEqual(f(2), 3)
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)
        g = PiecewiseLinearFunction([1, 1, 2, 3, 4], [1, 2, 3, 4, 5], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 2.5)
        self.assertEqual(f(2), 3)
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        self.assertEqual(f(3.5), 4.5)
        self.assertEqual(f(4), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)
        g = PiecewiseLinearFunction([1, 2, 2, 3, 4], [1, 2, 3, 4, 5], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        self.assertEqual(f(3.5), 4.5)
        self.assertEqual(f(4), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)
        g = PiecewiseLinearFunction([1, 2, 3, 3, 4], [1, 2, 3, 4, 5], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3)
        self.assertEqual(f(3.5), 4.5)
        self.assertEqual(f(4), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)
        g = PiecewiseLinearFunction([1, 2, 3, 4, 4], [1, 2, 3, 4, 5], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3)
        self.assertEqual(f(3.5), 3.5)
        self.assertEqual(f(4), 4)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)
        g = PiecewiseLinearFunction([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(g, require_bounded_input_variable=False, equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3)
        self.assertEqual(f(3.5), 3.5)
        self.assertEqual(f(4), 4)
        self.assertEqual(f(4.5), 4.5)
        self.assertEqual(f(5), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(5.1)

    def test_type(self):
        for key in transforms.registered_transforms:
            p = transforms.piecewise([1, 2, 3], [1, 2, 1], repn=key, validate=False)
            self.assertTrue(len(list(p.children())) <= 4)
            self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunction))
            self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
            self.assertTrue(isinstance(p, ICategorizedObject))
            self.assertTrue(isinstance(p, ICategorizedObjectContainer))
            self.assertTrue(isinstance(p, IHeterogeneousContainer))
            self.assertTrue(isinstance(p, IBlock))
            self.assertTrue(isinstance(p, block))

    def test_bad_repn(self):
        repn = list(transforms.registered_transforms.keys())[0]
        self.assertTrue(repn in transforms.registered_transforms)
        transforms.piecewise([1, 2, 3], [1, 2, 1], validate=False, repn=repn)
        repn = '_bad_repn_'
        self.assertFalse(repn in transforms.registered_transforms)
        with self.assertRaises(ValueError):
            transforms.piecewise([1, 2, 3], [1, 2, 1], validate=False, repn=repn)
        with self.assertRaises(ValueError):
            transforms.piecewise([1, 2, 3], [1, 2, 1], input=variable(lb=1, ub=3), validate=True, simplify=False, repn=repn)
        with self.assertRaises(ValueError):
            transforms.piecewise([1, 2, 3], [1, 2, 1], input=variable(lb=1, ub=3), validate=True, simplify=True, repn=repn)

    def test_init(self):
        for key in transforms.registered_transforms:
            for bound in ['lb', 'ub', 'eq', 'bad']:
                for args in [([1, 2, 3], [1, 2, 1]), ([1, 2, 3, 4, 5], [1, 2, 1, 2, 1]), ([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 1, 2, 1, 2, 1, 2, 1])]:
                    kwds = {'repn': key, 'bound': bound, 'validate': False}
                    if bound == 'bad':
                        with self.assertRaises(ValueError):
                            transforms.piecewise(*args, **kwds)
                        kwds['simplify'] = True
                        with self.assertRaises(ValueError):
                            transforms.piecewise(*args, **kwds)
                        kwds['simplify'] = False
                        with self.assertRaises(ValueError):
                            transforms.piecewise(*args, **kwds)
                    else:
                        p = transforms.piecewise(*args, **kwds)
                        self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
                        self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunction))
                        self.assertEqual(p.active, True)
                        self.assertIs(p.parent, None)
                        kwds['simplify'] = True
                        p = transforms.piecewise(*args, **kwds)
                        self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
                        self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunction))
                        self.assertEqual(p.active, True)
                        self.assertIs(p.parent, None)
                        kwds['simplify'] = False
                        p = transforms.piecewise(*args, **kwds)
                        self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
                        self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunction))
                        self.assertEqual(p.active, True)
                        self.assertIs(p.parent, None)

    def test_bad_init(self):
        with self.assertRaises(ValueError):
            PiecewiseLinearFunction([1, 2, 3], [1, 2, 1, 1], validate=False)
        with self.assertRaises(ValueError):
            PiecewiseLinearFunction([1, 2, 3, 4], [1, 2, 1], validate=False)
        with self.assertRaises(util.PiecewiseValidationError):
            PiecewiseLinearFunction([1, 3, 2], [1, 2, 1])
        PiecewiseLinearFunction([1, 3, 2], [1, 2, 1], validate=False)
        PiecewiseLinearFunction([1, 2, 3], [1, 1, 1 + 2e-06], equal_slopes_tolerance=1e-06)
        with self.assertRaises(util.PiecewiseValidationError):
            PiecewiseLinearFunction([1, 2, 3], [1, 1, 1 + 2e-06], equal_slopes_tolerance=3e-06)
        PiecewiseLinearFunction([1, 2, 3], [1, 1, 1 + 2e-06], validate=False)
        f = PiecewiseLinearFunction([1, 2, 3], [1, 2, 1])
        TransformedPiecewiseLinearFunction(f, input=variable(lb=1, ub=3), require_bounded_input_variable=True)
        TransformedPiecewiseLinearFunction(f, input=variable(lb=1, ub=3), require_bounded_input_variable=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f, input=variable(lb=1), require_bounded_input_variable=True)
        TransformedPiecewiseLinearFunction(f, input=variable(lb=1), require_bounded_input_variable=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f, input=variable(ub=3), require_bounded_input_variable=True)
        TransformedPiecewiseLinearFunction(f, input=variable(ub=3), require_bounded_input_variable=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f, require_bounded_input_variable=True)
        TransformedPiecewiseLinearFunction(f, require_bounded_input_variable=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f, input=variable(lb=0), require_bounded_input_variable=False, require_variable_domain_coverage=True)
        TransformedPiecewiseLinearFunction(f, input=variable(lb=0), require_bounded_input_variable=False, require_variable_domain_coverage=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f, input=variable(ub=4), require_bounded_input_variable=False, require_variable_domain_coverage=True)
        TransformedPiecewiseLinearFunction(f, input=variable(ub=4), require_bounded_input_variable=False, require_variable_domain_coverage=False)

    def test_bad_init_log_types(self):
        with self.assertRaises(ValueError):
            transforms.piecewise([1, 2, 3, 4], [1, 2, 3, 4], repn='dlog', validate=False)
        with self.assertRaises(ValueError):
            transforms.piecewise([1, 2, 3, 4], [1, 2, 3, 4], repn='log', validate=False)

    def test_step(self):
        breakpoints = [1, 2, 2]
        values = [1, 0, 1]
        v = variable()
        v.bounds = (min(breakpoints), max(breakpoints))
        for key in transforms.registered_transforms:
            if key in ('mc', 'convex'):
                with self.assertRaises(util.PiecewiseValidationError):
                    transforms.piecewise(breakpoints, values, input=v, repn=key)
            else:
                p = transforms.piecewise(breakpoints, values, input=v, repn=key)
                self.assertEqual(p.validate(), 4)

    def test_simplify(self):
        v = variable(lb=1, ub=3)
        convex_breakpoints = [1, 2, 3]
        convex_values = [1, 0, 1]
        for key in transforms.registered_transforms:
            for bound in ('lb', 'ub', 'eq'):
                if key == 'convex' and bound != 'lb':
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(convex_breakpoints, convex_values, input=v, repn=key, bound=bound, simplify=False)
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(convex_breakpoints, convex_values, input=v, repn=key, bound=bound, simplify=True)
                else:
                    p = transforms.piecewise(convex_breakpoints, convex_values, input=v, repn=key, bound=bound, simplify=False)
                    self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
                    self.assertEqual(p.validate(), util.characterize_function.convex)
                    p = transforms.piecewise(convex_breakpoints, convex_values, input=v, repn=key, bound=bound, simplify=True)
                    if bound == 'lb':
                        self.assertTrue(isinstance(p, transforms.registered_transforms['convex']))
                    else:
                        self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
        concave_breakpoints = [1, 2, 3]
        concave_values = [-1, 0, -1]
        for key in transforms.registered_transforms:
            for bound in ('lb', 'ub', 'eq'):
                if key == 'convex' and bound != 'ub':
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(concave_breakpoints, concave_values, input=v, repn=key, bound=bound, simplify=False)
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(concave_breakpoints, concave_values, input=v, repn=key, bound=bound, simplify=True)
                else:
                    p = transforms.piecewise(concave_breakpoints, concave_values, input=v, repn=key, bound=bound, simplify=False)
                    self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
                    self.assertEqual(p.validate(), util.characterize_function.concave)
                    p = transforms.piecewise(concave_breakpoints, concave_values, input=v, repn=key, bound=bound, simplify=True)
                    if bound == 'ub':
                        self.assertTrue(isinstance(p, transforms.registered_transforms['convex']))
                    else:
                        self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
        affine_breakpoints = [1, 3]
        affine_values = [1, 3]
        for key in transforms.registered_transforms:
            for bound in ('lb', 'ub', 'eq'):
                p = transforms.piecewise(affine_breakpoints, affine_values, input=v, repn=key, bound=bound, simplify=False)
                self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
                self.assertEqual(p.validate(), util.characterize_function.affine)
                p = transforms.piecewise(affine_breakpoints, affine_values, input=v, repn=key, bound=bound, simplify=True)
                self.assertTrue(isinstance(p, transforms.registered_transforms['convex']))