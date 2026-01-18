from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
class FireDecoratorsTest(testutils.BaseTestCase):

    def testSetParseFnsNamedArgs(self):
        self.assertEqual(core.Fire(NoDefaults, command=['double', '2']), 4)
        self.assertEqual(core.Fire(NoDefaults, command=['triple', '4']), 12.0)

    def testSetParseFnsPositionalArgs(self):
        self.assertEqual(core.Fire(NoDefaults, command=['quadruple', '5']), 20)

    def testSetParseFnsFnWithPositionalArgs(self):
        self.assertEqual(core.Fire(double, command=['5']), 10)

    def testSetParseFnsDefaultsFromPython(self):
        self.assertTupleEqual(WithDefaults().example1(), (10, int))
        self.assertEqual(WithDefaults().example1(5), (5, int))
        self.assertEqual(WithDefaults().example1(12.0), (12, float))

    def testSetParseFnsDefaultsFromFire(self):
        self.assertEqual(core.Fire(WithDefaults, command=['example1']), (10, int))
        self.assertEqual(core.Fire(WithDefaults, command=['example1', '10']), (10, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example1', '13']), (13, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example1', '14.0']), (14, float))

    def testSetParseFnsNamedDefaultsFromPython(self):
        self.assertTupleEqual(WithDefaults().example2(), (10, int))
        self.assertEqual(WithDefaults().example2(5), (5, int))
        self.assertEqual(WithDefaults().example2(12.0), (12, float))

    def testSetParseFnsNamedDefaultsFromFire(self):
        self.assertEqual(core.Fire(WithDefaults, command=['example2']), (10, int))
        self.assertEqual(core.Fire(WithDefaults, command=['example2', '10']), (10, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example2', '13']), (13, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example2', '14.0']), (14, float))

    def testSetParseFnsPositionalAndNamed(self):
        self.assertEqual(core.Fire(MixedArguments, ['example3', '10', '10']), (10, '10'))

    def testSetParseFnsOnlySomeTypes(self):
        self.assertEqual(core.Fire(PartialParseFn, command=['example4', '10', '10']), ('10', 10))
        self.assertEqual(core.Fire(PartialParseFn, command=['example5', '10', '10']), (10, '10'))

    def testSetParseFnsForKeywordArgs(self):
        self.assertEqual(core.Fire(WithKwargs, command=['example6']), ('default', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--herring', '"red"']), ('default', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', 'train']), ('train', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', '3']), ('3', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', '-1', '--count', '10']), ('-1', 10))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--count', '-2']), ('default', -2))

    def testSetParseFn(self):
        self.assertEqual(core.Fire(WithVarArgs, command=['example7', '1', '--arg2=2', '3', '4', '--kwarg=5']), ('1', '2', ('3', '4'), {'kwarg': '5'}))