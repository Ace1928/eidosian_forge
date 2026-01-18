from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
class PydevTestRunner(object):
    """ finds and runs a file or directory of files as a unit test """
    __py_extensions = ['*.py', '*.pyw']
    __exclude_files = ['__init__.*']
    __slots__ = ['verbosity', 'files_to_tests', 'files_or_dirs', 'include_tests', 'tests', 'jobs', 'split_jobs', 'configuration', 'coverage']

    def __init__(self, configuration):
        self.verbosity = configuration.verbosity
        self.jobs = configuration.jobs
        self.split_jobs = configuration.split_jobs
        files_to_tests = configuration.files_to_tests
        if files_to_tests:
            self.files_to_tests = files_to_tests
            self.files_or_dirs = list(files_to_tests.keys())
            self.tests = None
        else:
            self.files_to_tests = {}
            self.files_or_dirs = configuration.files_or_dirs
            self.tests = configuration.tests
        self.configuration = configuration
        self.__adjust_path()

    def __adjust_path(self):
        """ add the current file or directory to the python path """
        path_to_append = None
        for n in range(len(self.files_or_dirs)):
            dir_name = self.__unixify(self.files_or_dirs[n])
            if os.path.isdir(dir_name):
                if not dir_name.endswith('/'):
                    self.files_or_dirs[n] = dir_name + '/'
                path_to_append = os.path.normpath(dir_name)
            elif os.path.isfile(dir_name):
                path_to_append = os.path.dirname(dir_name)
            else:
                if not os.path.exists(dir_name):
                    block_line = '*' * 120
                    sys.stderr.write('\n%s\n* PyDev test runner error: %s does not exist.\n%s\n' % (block_line, dir_name, block_line))
                    return
                msg = 'unknown type. \n%s\nshould be file or a directory.\n' % dir_name
                raise RuntimeError(msg)
        if path_to_append is not None:
            sys.path.append(path_to_append)

    def __is_valid_py_file(self, fname):
        """ tests that a particular file contains the proper file extension
            and is not in the list of files to exclude """
        is_valid_fname = 0
        for invalid_fname in self.__class__.__exclude_files:
            is_valid_fname += int(not fnmatch.fnmatch(fname, invalid_fname))
        if_valid_ext = 0
        for ext in self.__class__.__py_extensions:
            if_valid_ext += int(fnmatch.fnmatch(fname, ext))
        return is_valid_fname > 0 and if_valid_ext > 0

    def __unixify(self, s):
        """ stupid windows. converts the backslash to forwardslash for consistency """
        return os.path.normpath(s).replace(os.sep, '/')

    def __importify(self, s, dir=False):
        """ turns directory separators into dots and removes the ".py*" extension
            so the string can be used as import statement """
        if not dir:
            dirname, fname = os.path.split(s)
            if fname.count('.') > 1:
                return
            imp_stmt_pieces = [dirname.replace('\\', '/').replace('/', '.'), os.path.splitext(fname)[0]]
            if len(imp_stmt_pieces[0]) == 0:
                imp_stmt_pieces = imp_stmt_pieces[1:]
            return '.'.join(imp_stmt_pieces)
        else:
            return s.replace('\\', '/').replace('/', '.')

    def __add_files(self, pyfiles, root, files):
        """ if files match, appends them to pyfiles. used by os.path.walk fcn """
        for fname in files:
            if self.__is_valid_py_file(fname):
                name_without_base_dir = self.__unixify(os.path.join(root, fname))
                pyfiles.append(name_without_base_dir)

    def find_import_files(self):
        """ return a list of files to import """
        if self.files_to_tests:
            pyfiles = self.files_to_tests.keys()
        else:
            pyfiles = []
            for base_dir in self.files_or_dirs:
                if os.path.isdir(base_dir):
                    for root, dirs, files in os.walk(base_dir):
                        exclude = {}
                        for d in dirs:
                            for init in ['__init__.py', '__init__.pyo', '__init__.pyc', '__init__.pyw', '__init__$py.class']:
                                if os.path.exists(os.path.join(root, d, init).replace('\\', '/')):
                                    break
                            else:
                                exclude[d] = 1
                        if exclude:
                            new = []
                            for d in dirs:
                                if d not in exclude:
                                    new.append(d)
                            dirs[:] = new
                        self.__add_files(pyfiles, root, files)
                elif os.path.isfile(base_dir):
                    pyfiles.append(base_dir)
        if self.configuration.exclude_files or self.configuration.include_files:
            ret = []
            for f in pyfiles:
                add = True
                basename = os.path.basename(f)
                if self.configuration.include_files:
                    add = False
                    for pat in self.configuration.include_files:
                        if fnmatch.fnmatchcase(basename, pat):
                            add = True
                            break
                if not add:
                    if self.verbosity > 3:
                        sys.stdout.write('Skipped file: %s (did not match any include_files pattern: %s)\n' % (f, self.configuration.include_files))
                elif self.configuration.exclude_files:
                    for pat in self.configuration.exclude_files:
                        if fnmatch.fnmatchcase(basename, pat):
                            if self.verbosity > 3:
                                sys.stdout.write('Skipped file: %s (matched exclude_files pattern: %s)\n' % (f, pat))
                            elif self.verbosity > 2:
                                sys.stdout.write('Skipped file: %s\n' % (f,))
                            add = False
                            break
                if add:
                    if self.verbosity > 3:
                        sys.stdout.write('Adding file: %s for test discovery.\n' % (f,))
                    ret.append(f)
            pyfiles = ret
        return pyfiles

    def __get_module_from_str(self, modname, print_exception, pyfile):
        """ Import the module in the given import path.
            * Returns the "final" module, so importing "coilib40.subject.visu"
            returns the "visu" module, not the "coilib40" as returned by __import__ """
        try:
            mod = __import__(modname)
            for part in modname.split('.')[1:]:
                mod = getattr(mod, part)
            return mod
        except:
            if print_exception:
                from _pydev_runfiles import pydev_runfiles_xml_rpc
                from _pydevd_bundle import pydevd_io
                buf_err = pydevd_io.start_redirect(keep_original_redirection=True, std='stderr')
                buf_out = pydevd_io.start_redirect(keep_original_redirection=True, std='stdout')
                try:
                    import traceback
                    traceback.print_exc()
                    sys.stderr.write('ERROR: Module: %s could not be imported (file: %s).\n' % (modname, pyfile))
                finally:
                    pydevd_io.end_redirect('stderr')
                    pydevd_io.end_redirect('stdout')
                pydev_runfiles_xml_rpc.notifyTest('error', buf_out.getvalue(), buf_err.getvalue(), pyfile, modname, 0)
            return None

    def remove_duplicates_keeping_order(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def find_modules_from_files(self, pyfiles):
        """ returns a list of modules given a list of files """
        imports = [(s, self.__importify(s)) for s in pyfiles]
        sys_path = [os.path.normpath(path) for path in sys.path]
        sys_path = self.remove_duplicates_keeping_order(sys_path)
        system_paths = []
        for s in sys_path:
            system_paths.append(self.__importify(s, True))
        ret = []
        for pyfile, imp in imports:
            if imp is None:
                continue
            choices = []
            for s in system_paths:
                if imp.startswith(s):
                    add = imp[len(s) + 1:]
                    if add:
                        choices.append(add)
            if not choices:
                sys.stdout.write('PYTHONPATH not found for file: %s\n' % imp)
            else:
                for i, import_str in enumerate(choices):
                    print_exception = i == len(choices) - 1
                    mod = self.__get_module_from_str(import_str, print_exception, pyfile)
                    if mod is not None:
                        ret.append((pyfile, mod, import_str))
                        break
        return ret

    class GetTestCaseNames:
        """Yes, we need a class for that (cannot use outer context on jython 2.1)"""

        def __init__(self, accepted_classes, accepted_methods):
            self.accepted_classes = accepted_classes
            self.accepted_methods = accepted_methods

        def __call__(self, testCaseClass):
            """Return a sorted sequence of method names found within testCaseClass"""
            testFnNames = []
            className = testCaseClass.__name__
            if className in self.accepted_classes:
                for attrname in dir(testCaseClass):
                    if attrname.startswith('test') and hasattr(getattr(testCaseClass, attrname), '__call__'):
                        testFnNames.append(attrname)
            else:
                for attrname in dir(testCaseClass):
                    if className + '.' + attrname in self.accepted_methods:
                        if hasattr(getattr(testCaseClass, attrname), '__call__'):
                            testFnNames.append(attrname)
            testFnNames.sort()
            return testFnNames

    def _decorate_test_suite(self, suite, pyfile, module_name):
        import unittest
        if isinstance(suite, unittest.TestSuite):
            add = False
            suite.__pydev_pyfile__ = pyfile
            suite.__pydev_module_name__ = module_name
            for t in suite._tests:
                t.__pydev_pyfile__ = pyfile
                t.__pydev_module_name__ = module_name
                if self._decorate_test_suite(t, pyfile, module_name):
                    add = True
            return add
        elif isinstance(suite, unittest.TestCase):
            return True
        else:
            return False

    def find_tests_from_modules(self, file_and_modules_and_module_name):
        """ returns the unittests given a list of modules """
        from _pydev_runfiles import pydev_runfiles_unittest
        import unittest
        unittest.TestLoader.suiteClass = pydev_runfiles_unittest.PydevTestSuite
        loader = unittest.TestLoader()
        ret = []
        if self.files_to_tests:
            for pyfile, m, module_name in file_and_modules_and_module_name:
                accepted_classes = {}
                accepted_methods = {}
                tests = self.files_to_tests[pyfile]
                for t in tests:
                    accepted_methods[t] = t
                loader.getTestCaseNames = self.GetTestCaseNames(accepted_classes, accepted_methods)
                suite = loader.loadTestsFromModule(m)
                if self._decorate_test_suite(suite, pyfile, module_name):
                    ret.append(suite)
            return ret
        if self.tests:
            accepted_classes = {}
            accepted_methods = {}
            for t in self.tests:
                splitted = t.split('.')
                if len(splitted) == 1:
                    accepted_classes[t] = t
                elif len(splitted) == 2:
                    accepted_methods[t] = t
            loader.getTestCaseNames = self.GetTestCaseNames(accepted_classes, accepted_methods)
        for pyfile, m, module_name in file_and_modules_and_module_name:
            suite = loader.loadTestsFromModule(m)
            if self._decorate_test_suite(suite, pyfile, module_name):
                ret.append(suite)
        return ret

    def filter_tests(self, test_objs, internal_call=False):
        """ based on a filter name, only return those tests that have
            the test case names that match """
        import unittest
        if not internal_call:
            if not self.configuration.include_tests and (not self.tests) and (not self.configuration.exclude_tests):
                return test_objs
            if self.verbosity > 1:
                if self.configuration.include_tests:
                    sys.stdout.write('Tests to include: %s\n' % (self.configuration.include_tests,))
                if self.tests:
                    sys.stdout.write('Tests to run: %s\n' % (self.tests,))
                if self.configuration.exclude_tests:
                    sys.stdout.write('Tests to exclude: %s\n' % (self.configuration.exclude_tests,))
        test_suite = []
        for test_obj in test_objs:
            if isinstance(test_obj, unittest.TestSuite):
                if test_obj._tests:
                    test_obj._tests = self.filter_tests(test_obj._tests, True)
                    if test_obj._tests:
                        test_suite.append(test_obj)
            elif isinstance(test_obj, unittest.TestCase):
                try:
                    testMethodName = test_obj._TestCase__testMethodName
                except AttributeError:
                    testMethodName = test_obj._testMethodName
                add = True
                if self.configuration.exclude_tests:
                    for pat in self.configuration.exclude_tests:
                        if fnmatch.fnmatchcase(testMethodName, pat):
                            if self.verbosity > 3:
                                sys.stdout.write('Skipped test: %s (matched exclude_tests pattern: %s)\n' % (testMethodName, pat))
                            elif self.verbosity > 2:
                                sys.stdout.write('Skipped test: %s\n' % (testMethodName,))
                            add = False
                            break
                if add:
                    if self.__match_tests(self.tests, test_obj, testMethodName):
                        include = True
                        if self.configuration.include_tests:
                            include = False
                            for pat in self.configuration.include_tests:
                                if fnmatch.fnmatchcase(testMethodName, pat):
                                    include = True
                                    break
                        if include:
                            test_suite.append(test_obj)
                        elif self.verbosity > 3:
                            sys.stdout.write('Skipped test: %s (did not match any include_tests pattern %s)\n' % (testMethodName, self.configuration.include_tests))
        return test_suite

    def iter_tests(self, test_objs):
        import unittest
        tests = []
        for test_obj in test_objs:
            if isinstance(test_obj, unittest.TestSuite):
                tests.extend(self.iter_tests(test_obj._tests))
            elif isinstance(test_obj, unittest.TestCase):
                tests.append(test_obj)
        return tests

    def list_test_names(self, test_objs):
        names = []
        for tc in self.iter_tests(test_objs):
            try:
                testMethodName = tc._TestCase__testMethodName
            except AttributeError:
                testMethodName = tc._testMethodName
            names.append(testMethodName)
        return names

    def __match_tests(self, tests, test_case, test_method_name):
        if not tests:
            return 1
        for t in tests:
            class_and_method = t.split('.')
            if len(class_and_method) == 1:
                if class_and_method[0] == test_case.__class__.__name__:
                    return 1
            elif len(class_and_method) == 2:
                if class_and_method[0] == test_case.__class__.__name__ and class_and_method[1] == test_method_name:
                    return 1
        return 0

    def __match(self, filter_list, name):
        """ returns whether a test name matches the test filter """
        if filter_list is None:
            return 1
        for f in filter_list:
            if re.match(f, name):
                return 1
        return 0

    def run_tests(self, handle_coverage=True):
        """ runs all tests """
        sys.stdout.write('Finding files... ')
        files = self.find_import_files()
        if self.verbosity > 3:
            sys.stdout.write('%s ... done.\n' % self.files_or_dirs)
        else:
            sys.stdout.write('done.\n')
        sys.stdout.write('Importing test modules ... ')
        if handle_coverage:
            coverage_files, coverage = start_coverage_support(self.configuration)
        file_and_modules_and_module_name = self.find_modules_from_files(files)
        sys.stdout.write('done.\n')
        all_tests = self.find_tests_from_modules(file_and_modules_and_module_name)
        all_tests = self.filter_tests(all_tests)
        from _pydev_runfiles import pydev_runfiles_unittest
        test_suite = pydev_runfiles_unittest.PydevTestSuite(all_tests)
        from _pydev_runfiles import pydev_runfiles_xml_rpc
        pydev_runfiles_xml_rpc.notifyTestsCollected(test_suite.countTestCases())
        start_time = time.time()

        def run_tests():
            executed_in_parallel = False
            if self.jobs > 1:
                from _pydev_runfiles import pydev_runfiles_parallel
                executed_in_parallel = pydev_runfiles_parallel.execute_tests_in_parallel(all_tests, self.jobs, self.split_jobs, self.verbosity, coverage_files, self.configuration.coverage_include)
            if not executed_in_parallel:
                runner = pydev_runfiles_unittest.PydevTextTestRunner(stream=sys.stdout, descriptions=1, verbosity=self.verbosity)
                sys.stdout.write('\n')
                runner.run(test_suite)
        if self.configuration.django:
            get_django_test_suite_runner()(run_tests).run_tests([])
        else:
            run_tests()
        if handle_coverage:
            coverage.stop()
            coverage.save()
        total_time = 'Finished in: %.2f secs.' % (time.time() - start_time,)
        pydev_runfiles_xml_rpc.notifyTestRunFinished(total_time)