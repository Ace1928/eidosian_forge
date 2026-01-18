import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
class ETSConfigTestCase(unittest.TestCase):
    """ Tests the 'ETSConfig' configuration object. """

    def setUp(self):
        """
        Prepares the test fixture before each test method is called.

        """
        self.ETSConfig = type(ETSConfig)()

    def run(self, result=None):
        with temporary_home_directory():
            super().run(result)

    def test_application_data(self):
        """
        application data

        """
        dirname = self.ETSConfig.application_data
        self.assertEqual(os.path.exists(dirname), True)
        self.assertEqual(os.path.isdir(dirname), True)

    def test_set_application_data(self):
        """
        set application data

        """
        old = self.ETSConfig.application_data
        self.ETSConfig.application_data = 'foo'
        self.assertEqual('foo', self.ETSConfig.application_data)
        self.ETSConfig.application_data = old
        self.assertEqual(old, self.ETSConfig.application_data)

    def test_application_data_is_idempotent(self):
        """
        application data is idempotent

        """
        self.test_application_data()
        self.test_application_data()

    def test_write_to_application_data_directory(self):
        """
        write to application data directory

        """
        self.ETSConfig.company = 'Blah'
        dirname = self.ETSConfig.application_data
        path = os.path.join(dirname, 'dummy.txt')
        data = str(time.time())
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
        self.assertEqual(os.path.exists(path), True)
        with open(path, 'r', encoding='utf-8') as f:
            result = f.read()
        os.remove(path)
        self.assertEqual(data, result)

    def test_default_company(self):
        """
        default company

        """
        self.assertEqual(self.ETSConfig.company, 'Enthought')

    def test_set_company(self):
        """
        set company

        """
        old = self.ETSConfig.company
        self.ETSConfig.company = 'foo'
        self.assertEqual('foo', self.ETSConfig.company)
        self.ETSConfig.company = old
        self.assertEqual(old, self.ETSConfig.company)

    def _test_default_application_home(self):
        """
        application home

        """
        app_home = self.ETSConfig.application_home
        dirname, app_name = os.path.split(app_home)
        self.assertEqual(dirname, self.ETSConfig.application_data)
        self.assertEqual(app_name, 'tests')

    def test_toolkit_default_kiva_backend(self):
        self.ETSConfig.toolkit = 'qt4'
        self.assertEqual(self.ETSConfig.kiva_backend, 'image')

    def test_default_backend_for_qt5_toolkit(self):
        self.ETSConfig.toolkit = 'qt'
        self.assertEqual(self.ETSConfig.kiva_backend, 'image')

    def test_toolkit_explicit_kiva_backend(self):
        self.ETSConfig.toolkit = 'wx.celiagg'
        self.assertEqual(self.ETSConfig.kiva_backend, 'celiagg')

    def test_toolkit_environ(self):
        test_args = ['something']
        test_environ = {'ETS_TOOLKIT': 'test'}
        with mock_sys_argv(test_args):
            with mock_os_environ(test_environ):
                toolkit = self.ETSConfig.toolkit
        self.assertEqual(toolkit, 'test')

    def test_toolkit_environ_missing(self):
        test_args = ['something']
        test_environ = {}
        with mock_sys_argv(test_args):
            with mock_os_environ(test_environ):
                toolkit = self.ETSConfig.toolkit
        self.assertEqual(toolkit, '')

    def test_set_toolkit(self):
        test_args = []
        test_environ = {'ETS_TOOLKIT': 'test_environ'}
        with mock_sys_argv(test_args):
            with mock_os_environ(test_environ):
                self.ETSConfig.toolkit = 'test_direct'
                toolkit = self.ETSConfig.toolkit
        self.assertEqual(toolkit, 'test_direct')

    def test_provisional_toolkit(self):
        test_args = []
        test_environ = {}
        with mock_sys_argv(test_args):
            with mock_os_environ(test_environ):
                repr(self.ETSConfig.toolkit)
                with self.ETSConfig.provisional_toolkit('test_direct'):
                    toolkit = self.ETSConfig.toolkit
                    self.assertEqual(toolkit, 'test_direct')
        toolkit = self.ETSConfig.toolkit
        self.assertEqual(toolkit, 'test_direct')

    def test_provisional_toolkit_exception(self):
        test_args = []
        test_environ = {'ETS_TOOLKIT': ''}
        with mock_sys_argv(test_args):
            with mock_os_environ(test_environ):
                try:
                    with self.ETSConfig.provisional_toolkit('test_direct'):
                        toolkit = self.ETSConfig.toolkit
                        self.assertEqual(toolkit, 'test_direct')
                        raise ETSToolkitError('Test exception')
                except ETSToolkitError as exc:
                    if not exc.message == 'Test exception':
                        raise
                toolkit = self.ETSConfig.toolkit
                self.assertEqual(toolkit, '')

    def test_provisional_toolkit_already_set(self):
        test_args = []
        test_environ = {'ETS_TOOLKIT': 'test_environ'}
        with mock_sys_argv(test_args):
            with mock_os_environ(test_environ):
                with self.assertRaises(ETSToolkitError):
                    with self.ETSConfig.provisional_toolkit('test_direct'):
                        pass
                toolkit = self.ETSConfig.toolkit
                self.assertEqual(toolkit, 'test_environ')

    def test_user_data(self):
        """
        user data

        """
        dirname = self.ETSConfig.user_data
        self.assertEqual(os.path.exists(dirname), True)
        self.assertEqual(os.path.isdir(dirname), True)

    def test_set_user_data(self):
        """
        set user data

        """
        old = self.ETSConfig.user_data
        self.ETSConfig.user_data = 'foo'
        self.assertEqual('foo', self.ETSConfig.user_data)
        self.ETSConfig.user_data = old
        self.assertEqual(old, self.ETSConfig.user_data)

    def test_user_data_is_idempotent(self):
        """
        user data is idempotent

        """
        self.test_user_data()

    def test_write_to_user_data_directory(self):
        """
        write to user data directory

        """
        self.ETSConfig.company = 'Blah'
        dirname = self.ETSConfig.user_data
        path = os.path.join(dirname, 'dummy.txt')
        data = str(time.time())
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
        self.assertEqual(os.path.exists(path), True)
        with open(path, 'r', encoding='utf-8') as f:
            result = f.read()
        os.remove(path)
        self.assertEqual(data, result)