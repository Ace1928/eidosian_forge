import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
class TestSystemCallSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ
        cls.oldpwd = os.getcwd()
        os.chdir(thisdir)
        try:
            assert exedir != exedir_user
            assert not os.path.exists(notexe_abspath_user)
            assert not os.path.exists(isexe_abspath_user)
            assert not os.path.exists(notexe_nopath)
            assert os.path.exists(os.path.join(exedir, notexe_nopath))
            assert os.path.exists(notexe_abspath)
            assert os.path.exists(notexe_relpath)
            assert not os.path.exists(isexe_nopath)
            assert os.path.exists(os.path.join(exedir, isexe_nopath))
            assert os.path.exists(isexe_abspath)
            assert os.path.exists(isexe_relpath)
            assert os.path.isfile(os.path.join(exedir, notexe_nopath))
            assert os.path.isfile(notexe_abspath)
            assert os.path.isfile(notexe_relpath)
            assert os.path.isfile(os.path.join(exedir, isexe_nopath))
            assert os.path.isfile(isexe_abspath)
            assert os.path.isfile(isexe_relpath)
            assert not os.access(os.path.join(exedir, notexe_nopath), os.X_OK)
            assert not os.access(notexe_abspath, os.X_OK)
            assert not os.access(notexe_relpath, os.X_OK)
            assert os.access(os.path.join(exedir, isexe_nopath), os.X_OK)
            assert os.access(isexe_abspath, os.X_OK)
            assert os.access(isexe_relpath, os.X_OK)
        except:
            os.chdir(cls.oldpwd)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.oldpwd)

    def setUp(self):
        os.chdir(thisdir)

    def test_noexe(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(NotImplementedError):
                opt.executable()

    def test_available(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(ApplicationError):
                opt.available(exception_flag=True)

    def test_reset_executable(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(NotImplementedError):
                opt.set_executable()
            opt._user_executable = 'x'
            opt.set_executable(validate=False)
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(NotImplementedError):
                opt.executable()

    def test_set_executable_notexe_nopath(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(ValueError):
                opt.set_executable(notexe_nopath)
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(notexe_nopath, validate=False)
            self.assertEqual(opt._user_executable, notexe_nopath)
            self.assertEqual(opt.executable(), notexe_nopath)

    @unittest.skipIf(is_windows, 'Skipping test because it requires testing if a file is executable on Windows')
    def test_set_executable_notexe_relpath(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(ValueError):
                opt.set_executable(notexe_relpath)
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(notexe_relpath, validate=False)
            self.assertEqual(opt._user_executable, notexe_relpath)
            self.assertEqual(opt.executable(), notexe_relpath)

    @unittest.skipIf(is_windows, 'Skipping test because it requires testing if a file is executable on Windows')
    def test_set_executable_notexe_abspath(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(ValueError):
                opt.set_executable(notexe_abspath)
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(notexe_abspath, validate=False)
            self.assertEqual(opt._user_executable, notexe_abspath)
            self.assertEqual(opt.executable(), notexe_abspath)

    @unittest.skipIf(is_windows, 'Skipping test because it requires testing if a file is executable on Windows')
    def test_set_executable_notexe_abspath_user(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(ValueError):
                opt.set_executable(notexe_abspath_user)
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(notexe_abspath_user, validate=False)
            self.assertEqual(opt._user_executable, notexe_abspath_user)
            self.assertEqual(opt.executable(), notexe_abspath_user)

    def test_set_executable_isexe_nopath(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            with self.assertRaises(ValueError):
                opt.set_executable(isexe_nopath)
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(isexe_nopath, validate=False)
            self.assertEqual(opt._user_executable, isexe_nopath)
            self.assertEqual(opt.executable(), isexe_nopath)
            opt._user_executable = None
            rm_PATH = False
            orig_PATH = None
            if 'PATH' in os.environ:
                orig_PATH = os.environ['PATH']
            else:
                rm_PATH = True
                os.environ['PATH'] = ''
            os.environ['PATH'] = exedir + os.pathsep + os.environ['PATH']
            try:
                opt.set_executable(isexe_nopath)
                self.assertEqual(opt._user_executable, isexe_abspath)
                self.assertEqual(opt.executable(), isexe_abspath)
            finally:
                if rm_PATH:
                    del os.environ['PATH']
                else:
                    os.environ['PATH'] = orig_PATH

    def test_set_executable_isexe_relpath(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(isexe_relpath)
            self.assertEqual(opt._user_executable, isexe_abspath)
            self.assertEqual(opt.executable(), isexe_abspath)
            opt._user_executable = None
            opt.set_executable(isexe_relpath, validate=False)
            self.assertEqual(opt._user_executable, isexe_relpath)
            self.assertEqual(opt.executable(), isexe_relpath)

    def test_set_executable_isexe_abspath(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(isexe_abspath)
            self.assertEqual(opt._user_executable, isexe_abspath)
            self.assertEqual(opt.executable(), isexe_abspath)
            opt._user_executable = None
            opt.set_executable(isexe_abspath, validate=False)
            self.assertEqual(opt._user_executable, isexe_abspath)
            self.assertEqual(opt.executable(), isexe_abspath)

    def test_set_executable_isexe_abspath_user(self):
        with SystemCallSolver(type='test') as opt:
            self.assertEqual(id(opt._user_executable), id(None))
            opt.set_executable(isexe_abspath_user)
            self.assertTrue(os.path.samefile(opt._user_executable, isexe_abspath))
            self.assertTrue(os.path.samefile(opt.executable(), isexe_abspath))
            opt._user_executable = None
            opt.set_executable(isexe_abspath_user, validate=False)
            self.assertEqual(opt._user_executable, isexe_abspath_user)
            self.assertEqual(opt.executable(), isexe_abspath_user)

    def test_SolverFactory_executable_isexe_nopath(self):
        for name in _test_names:
            with SolverFactory(name, executable=isexe_nopath) as opt:
                self.assertTrue(isinstance(opt, UnknownSolver))
        rm_PATH = False
        orig_PATH = None
        if 'PATH' in os.environ:
            orig_PATH = os.environ['PATH']
        else:
            rm_PATH = True
            os.environ['PATH'] = ''
        os.environ['PATH'] = exedir + os.pathsep + os.environ['PATH']
        try:
            for name in _test_names:
                with SolverFactory(name, executable=isexe_nopath) as opt:
                    if isinstance(opt, UnknownSolver):
                        continue
                    self.assertEqual(opt._user_executable, isexe_abspath)
                    self.assertEqual(opt.executable(), isexe_abspath)
        finally:
            if rm_PATH:
                del os.environ['PATH']
            else:
                os.environ['PATH'] = orig_PATH

    def test_SolverFactory_executable_isexe_relpath(self):
        for name in _test_names:
            with SolverFactory(name, executable=isexe_relpath) as opt:
                if isinstance(opt, UnknownSolver):
                    continue
                self.assertEqual(opt._user_executable, isexe_abspath)
                self.assertEqual(opt.executable(), isexe_abspath)

    def test_executable_isexe_abspath(self):
        for name in _test_names:
            with SolverFactory(name, executable=isexe_abspath) as opt:
                if isinstance(opt, UnknownSolver):
                    continue
                self.assertEqual(opt._user_executable, isexe_abspath)
                self.assertEqual(opt.executable(), isexe_abspath)

    def test_executable_isexe_abspath_user(self):
        for name in _test_names:
            with SolverFactory(name, executable=isexe_abspath_user) as opt:
                if isinstance(opt, UnknownSolver):
                    continue
                self.assertTrue(os.path.samefile(opt._user_executable, isexe_abspath))
                self.assertTrue(os.path.samefile(opt.executable(), isexe_abspath))