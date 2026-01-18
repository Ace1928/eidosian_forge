import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
class CPLEX_utils(unittest.TestCase):

    def test_validate_file_name(self):
        _126 = _mock_cplex_126()
        _128 = _mock_cplex_128()
        fname = 'foo.lp'
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))
        fname = 'foo bar.lp'
        with self.assertRaisesRegex(ValueError, 'Space detected in CPLEX xxx file'):
            _validate_file_name(_126, fname, 'xxx')
        self.assertEqual('"%s"' % (fname,), _validate_file_name(_128, fname, 'xxx'))
        fname = 'foo%sbar.lp' % (os.path.sep,)
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))
        bad_char = '/\\'.replace(os.path.sep, '')
        fname = 'foo%sbar.lp' % (bad_char,)
        msg = 'Unallowed character \\(%s\\) found in CPLEX xxx file' % (repr(bad_char)[1:-1],)
        with self.assertRaisesRegex(ValueError, msg):
            _validate_file_name(_126, fname, 'xxx')
        with self.assertRaisesRegex(ValueError, msg):
            _validate_file_name(_128, fname, 'xxx')
        fname = 'foo$$bar.lp'
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))
        fname = 'foo_bar.lp'
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))
        fname = 'foo&bar.lp'
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))
        fname = 'foo~bar.lp'
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))
        fname = 'foo-bar.lp'
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))
        bad_char = '^'
        fname = 'foo%sbar.lp' % (bad_char,)
        msg = 'Unallowed character \\(\\^\\) found in CPLEX xxx file'
        with self.assertRaisesRegex(ValueError, msg):
            _validate_file_name(_126, fname, 'xxx')
        with self.assertRaisesRegex(ValueError, msg):
            _validate_file_name(_128, fname, 'xxx')