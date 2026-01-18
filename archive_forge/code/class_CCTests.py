import os
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.mpec import Complementarity, complements, ComplementarityList
from pyomo.opt import ProblemFormat
from pyomo.repn.plugins.nl_writer import FileDeterminism
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
class CCTests(object):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def _setup(self):
        M = ConcreteModel()
        M.y = Var()
        M.x1 = Var()
        M.x2 = Var()
        M.x3 = Var()
        return M

    def _print(self, model):
        model.cc.pprint()

    def _test(self, tname, M):
        bfile = os.path.join(currdir, tname + f'_{self.xfrm}.txt')
        if self.xfrm is not None:
            xfrm = TransformationFactory(self.xfrm)
            xfrm.apply_to(M)
        with TempfileManager:
            ofile = TempfileManager.create_tempfile(suffix=f'_{self.xfrm}.out')
            with capture_output(ofile):
                self._print(M)
            try:
                self.assertTrue(cmp(ofile, bfile), msg='Files %s and %s differ' % (ofile, bfile))
            except:
                with open(ofile, 'r') as f1, open(bfile, 'r') as f2:
                    f1_contents = list(filter(None, f1.read().split()))
                    f2_contents = list(filter(None, f2.read().split()))
                    self.assertEqual(f1_contents, f2_contents)

    def test_t1a(self):
        M = self._setup()
        M.c = Constraint(expr=M.y + M.x3 >= M.x2)
        M.cc = Complementarity(expr=complements(M.y + M.x1 >= 0, M.x1 + 2 * M.x2 + 3 * M.x3 >= 1))
        self._test('t1a', M)

    def test_t1b(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + 2 * M.x2 + 3 * M.x3 >= 1, M.y + M.x1 >= 0))
        self._test('t1b', M)

    def test_t1c(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y >= -M.x1, M.x1 + 2 * M.x2 >= 1 - 3 * M.x3))
        self._test('t1c', M)

    def test_t2a(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x2 >= 0, M.x2 - M.x3 <= -1))
        self._test('t2a', M)

    def test_t2b(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x2 - M.x3 <= -1, M.y + M.x2 >= 0))
        self._test('t2b', M)

    def test_t3a(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x3 >= 0, M.x1 + M.x2 >= -1))
        self._test('t3a', M)

    def test_t3b(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + M.x2 >= -1, M.y + M.x3 >= 0))
        self._test('t3b', M)

    def test_t4a(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + 2 * M.x2 + 3 * M.x3 == 1, M.y + M.x3))
        self._test('t4a', M)

    def test_t4b(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x3, M.x1 + 2 * M.x2 + 3 * M.x3 == 1))
        self._test('t4b', M)

    def test_t4c(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(1 == M.x1 + 2 * M.x2 + 3 * M.x3, M.y + M.x3))
        self._test('t4c', M)

    def test_t4d(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + 2 * M.x2 == 1 - 3 * M.x3, M.y + M.x3))
        self._test('t4d', M)

    def test_t9(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x3, M.x1 + 2 * M.x2 == 1))
        M.cc.deactivate()
        M.keep_var_con = Constraint(expr=M.x1 == 0.5)
        self._test('t9', M)

    def test_t10(self):
        M = self._setup()

        def f(model, i):
            return complements(M.y + M.x3, M.x1 + 2 * M.x2 == i)
        M.cc = Complementarity([0, 1, 2], rule=f)
        M.cc[1].deactivate()
        self._test('t10', M)

    def test_t11(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(inequality(2, M.y + M.x1, 3), M.x1))
        self._test('t11', M)

    def test_t12(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1, inequality(2, M.y + M.x1, 3)))
        self._test('t12', M)

    def test_t13(self):
        M = self._setup()

        def f(model, i):
            if i == 0:
                return complements(M.y + M.x3, M.x1 + 2 * M.x2 == 0)
            if i == 1:
                return Complementarity.Skip
            if i == 2:
                return complements(M.y + M.x3, M.x1 + 2 * M.x2 == 2)
        M.cc = Complementarity([0, 1, 2], rule=f)
        self._test('t13', M)

    def test_cov2(self):
        M = self._setup()
        M.cc = Complementarity([0, 1, 2])
        M.keep_var_con = Constraint(expr=M.x1 == 0.5)
        self._test('cov2', M)

    def test_cov4(self):
        M = self._setup()

        def f(model):
            return complements(M.y + M.x3, M.x1 + 2 * M.x2 == 1)
        M.cc = Complementarity(rule=f)
        self._test('cov4', M)

    def test_cov5(self):
        M = self._setup()

        def f(model):
            raise IOError('cov5 error')
        try:
            M.cc1 = Complementarity(rule=f)
            self.fail('Expected an IOError')
        except IOError:
            pass

        def f(model, i):
            raise IOError('cov5 error')
        try:
            M.cc2 = Complementarity([0, 1], rule=f)
            self.fail('Expected an IOError')
        except IOError:
            pass

    def test_cov6(self):
        M = self._setup()
        with self.assertRaisesRegex(ValueError, 'Invalid tuple for Complementarity'):
            M.cc = Complementarity([0, 1], expr=())

    def test_cov7(self):
        M = self._setup()

        def f(model):
            return ()
        try:
            M.cc = Complementarity(rule=f)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        def f(model):
            return
        try:
            M.cc = Complementarity(rule=f)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        def f(model):
            return {}
        try:
            M.cc = Complementarity(rule=f)
            self.fail('Expected ValueError')
        except ValueError:
            pass

    def test_cov8(self):
        M = self._setup()

        def f(model):
            return [M.y + M.x3, M.x1 + 2 * M.x2 == 1]
        M.cc = Complementarity(rule=f)
        self._test('cov8', M)

    def test_cov9(self):
        M = self._setup()

        def f(model):
            return (M.y + M.x3, M.x1 + 2 * M.x2 == 1)
        M.cc = Complementarity(rule=f)
        self._test('cov8', M)

    def test_cov10(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(inequality(M.y, M.x1, 1), M.x2))
        try:
            M.cc.to_standard_form()
            self.fail('Expected a RuntimeError')
        except RuntimeError:
            pass

    def test_cov11(self):
        M = self._setup()
        M.cc = Complementarity(expr=complements(inequality(1, M.x1, M.y), M.x2))
        try:
            M.cc.to_standard_form()
            self.fail('Expected a RuntimeError')
        except RuntimeError:
            pass

    def test_list1(self):
        M = self._setup()
        M.cc = ComplementarityList()
        M.cc.add(complements(M.y + M.x3, M.x1 + 2 * M.x2 == 0))
        M.cc.add(complements(M.y + M.x3, M.x1 + 2 * M.x2 == 2))
        self._test('list1', M)

    def test_list2(self):
        M = self._setup()
        M.cc = ComplementarityList()
        M.cc.add(complements(M.y + M.x3, M.x1 + 2 * M.x2 == 0))
        M.cc.add(complements(M.y + M.x3, M.x1 + 2 * M.x2 == 1))
        M.cc.add(complements(M.y + M.x3, M.x1 + 2 * M.x2 == 2))
        M.cc[2].deactivate()
        self._test('list2', M)

    def test_list3(self):
        M = self._setup()

        def f(M, i):
            if i == 1:
                return complements(M.y + M.x3, M.x1 + 2 * M.x2 == 0)
            elif i == 2:
                return complements(M.y + M.x3, M.x1 + 2 * M.x2 == 2)
            return ComplementarityList.End
        M.cc = ComplementarityList(rule=f)
        self._test('list1', M)

    def test_list4(self):
        M = self._setup()

        def f(M):
            yield complements(M.y + M.x3, M.x1 + 2 * M.x2 == 0)
            yield complements(M.y + M.x3, M.x1 + 2 * M.x2 == 2)
            yield ComplementarityList.End
        M.cc = ComplementarityList(rule=f)
        self._test('list1', M)

    def test_list5(self):
        M = self._setup()
        M.cc = ComplementarityList(rule=(complements(M.y + M.x3, M.x1 + 2 * M.x2 == i) for i in range(3)))
        self._test('list5', M)

    def test_list6(self):
        M = self._setup()
        try:
            M.cc = ComplementarityList()
            self.fail('Expected a RuntimeError')
        except:
            pass

    def test_list7(self):
        M = self._setup()

        def f(M):
            return None
        try:
            M.cc = ComplementarityList(rule=f)
            self.fail('Expected a ValueError')
        except:
            pass
        M = self._setup()

        def f(M):
            yield None
        try:
            M.cc = ComplementarityList(rule=f)
            self.fail('Expected a ValueError')
        except:
            pass