import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptInterface(unittest.TestCase):

    def test_class_member_list(self):
        opt = ipopt.Ipopt()
        expected_list = ['Availability', 'CONFIG', 'config', 'available', 'is_persistent', 'solve', 'version', 'name']
        method_list = [method for method in dir(opt) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_instantiation(self):
        opt = ipopt.Ipopt()
        self.assertFalse(opt.is_persistent())
        self.assertIsNotNone(opt.version())
        self.assertEqual(opt.name, 'ipopt')
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertTrue(opt.available())

    def test_context_manager(self):
        with ipopt.Ipopt() as opt:
            self.assertFalse(opt.is_persistent())
            self.assertIsNotNone(opt.version())
            self.assertEqual(opt.name, 'ipopt')
            self.assertEqual(opt.CONFIG, opt.config)
            self.assertTrue(opt.available())

    def test_available_cache(self):
        opt = ipopt.Ipopt()
        opt.available()
        self.assertTrue(opt._available_cache[1])
        self.assertIsNotNone(opt._available_cache[0])
        config = ipopt.IpoptConfig()
        config.executable = Executable('/a/bogus/path')
        opt.available(config=config)
        self.assertFalse(opt._available_cache[1])
        self.assertIsNone(opt._available_cache[0])

    def test_version_cache(self):
        opt = ipopt.Ipopt()
        opt.version()
        self.assertIsNotNone(opt._version_cache[0])
        self.assertIsNotNone(opt._version_cache[1])
        config = ipopt.IpoptConfig()
        config.executable = Executable('/a/bogus/path')
        opt.version(config=config)
        self.assertIsNone(opt._version_cache[0])
        self.assertIsNone(opt._version_cache[1])

    def test_write_options_file(self):
        opt = ipopt.Ipopt()
        result = opt._write_options_file('fakename', None)
        self.assertFalse(result)
        opt = ipopt.Ipopt(solver_options={'max_iter': 4})
        result = opt._write_options_file('myfile', opt.config.solver_options)
        self.assertFalse(result)
        self.assertFalse(os.path.isfile('myfile.opt'))
        opt = ipopt.Ipopt(solver_options={'custom_option': 4})
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            if not os.path.exists(dname):
                os.mkdir(dname)
            filename = os.path.join(dname, 'myfile')
            result = opt._write_options_file(filename, opt.config.solver_options)
            self.assertTrue(result)
            self.assertTrue(os.path.isfile(filename + '.opt'))
        opt = ipopt.Ipopt(solver_options={'custom_option_1': 4, 'custom_option_2': 3})
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            if not os.path.exists(dname):
                os.mkdir(dname)
            filename = os.path.join(dname, 'myfile')
            result = opt._write_options_file(filename, opt.config.solver_options)
            self.assertTrue(result)
            self.assertTrue(os.path.isfile(filename + '.opt'))
            with open(filename + '.opt', 'r') as f:
                data = f.readlines()
                self.assertEqual(len(data), len(list(opt.config.solver_options.keys())))

    def test_create_command_line(self):
        opt = ipopt.Ipopt()
        result = opt._create_command_line('myfile', opt.config, False)
        self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL'])
        opt = ipopt.Ipopt(solver_options={'max_iter': 4})
        result = opt._create_command_line('myfile', opt.config, False)
        self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_iter=4'])
        opt = ipopt.Ipopt(solver_options={'max_iter': 4}, time_limit=10)
        result = opt._create_command_line('myfile', opt.config, False)
        self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_iter=4', 'max_cpu_time=10.0'])
        opt = ipopt.Ipopt(solver_options={'max_iter': 4, 'max_cpu_time': 10})
        result = opt._create_command_line('myfile', opt.config, False)
        self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_cpu_time=10', 'max_iter=4'])
        result = opt._create_command_line('myfile', opt.config, True)
        self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'option_file_name=myfile.opt', 'max_cpu_time=10', 'max_iter=4'])
        opt = ipopt.Ipopt(solver_options={'max_iter': 4, 'option_file_name': 'myfile.opt'})
        with self.assertRaises(ValueError):
            result = opt._create_command_line('myfile', opt.config, False)