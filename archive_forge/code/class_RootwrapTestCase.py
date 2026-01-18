import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
class RootwrapTestCase(testtools.TestCase):
    if os.path.exists('/sbin/ip'):
        _ip = '/sbin/ip'
    else:
        _ip = '/bin/ip'

    def setUp(self):
        super(RootwrapTestCase, self).setUp()
        self.filters = [filters.RegExpFilter('/bin/ls', 'root', 'ls', '/[a-z]+'), filters.CommandFilter('/usr/bin/foo_bar_not_exist', 'root'), filters.RegExpFilter('/bin/cat', 'root', 'cat', '/[a-z]+'), filters.CommandFilter('/nonexistent/cat', 'root'), filters.CommandFilter('/bin/cat', 'root')]

    def test_CommandFilter(self):
        f = filters.CommandFilter('sleep', 'root', '10')
        self.assertFalse(f.match(['sleep2']))
        self.assertTrue(f.match(['sleep']))
        self.assertTrue(f.match(['sleep', 'anything']))
        self.assertTrue(f.match(['sleep', '10']))
        f = filters.CommandFilter('sleep', 'root')
        self.assertTrue(f.match(['sleep', '10']))

    def test_empty_commandfilter(self):
        f = filters.CommandFilter('sleep', 'root')
        self.assertFalse(f.match([]))
        self.assertFalse(f.match(None))

    def test_empty_regexpfilter(self):
        f = filters.RegExpFilter('sleep', 'root', 'sleep')
        self.assertFalse(f.match([]))
        self.assertFalse(f.match(None))

    def test_empty_invalid_regexpfilter(self):
        f = filters.RegExpFilter('sleep', 'root')
        self.assertFalse(f.match(['anything']))
        self.assertFalse(f.match([]))

    def test_RegExpFilter_match(self):
        usercmd = ['ls', '/root']
        filtermatch = wrapper.match_filter(self.filters, usercmd)
        self.assertFalse(filtermatch is None)
        self.assertEqual(['/bin/ls', '/root'], filtermatch.get_command(usercmd))

    def test_RegExpFilter_reject(self):
        usercmd = ['ls', 'root']
        self.assertRaises(wrapper.NoFilterMatched, wrapper.match_filter, self.filters, usercmd)

    def test_missing_command(self):
        valid_but_missing = ['foo_bar_not_exist']
        invalid = ['foo_bar_not_exist_and_not_matched']
        self.assertRaises(wrapper.FilterMatchNotExecutable, wrapper.match_filter, self.filters, valid_but_missing)
        self.assertRaises(wrapper.NoFilterMatched, wrapper.match_filter, self.filters, invalid)

    def _test_EnvFilter_as_DnsMasq(self, config_file_arg):
        usercmd = ['env', config_file_arg + '=A', 'NETWORK_ID=foobar', 'dnsmasq', 'foo']
        f = filters.EnvFilter('env', 'root', config_file_arg + '=A', 'NETWORK_ID=', '/usr/bin/dnsmasq')
        self.assertTrue(f.match(usercmd))
        self.assertEqual(['/usr/bin/dnsmasq', 'foo'], f.get_command(usercmd))
        env = f.get_environment(usercmd)
        self.assertEqual('A', env.get(config_file_arg))
        self.assertEqual('foobar', env.get('NETWORK_ID'))

    def test_EnvFilter(self):
        envset = ['A=/some/thing', 'B=somethingelse']
        envcmd = ['env'] + envset
        realcmd = ['sleep', '10']
        usercmd = envcmd + realcmd
        f = filters.EnvFilter('env', 'root', 'A=', 'B=ignored', 'sleep')
        self.assertTrue(f.match(envcmd + ['sleep']))
        self.assertTrue(f.match(envset + ['sleep']))
        self.assertFalse(f.match(envcmd + ['sleep2']))
        self.assertFalse(f.match(envset + ['sleep2']))
        self.assertTrue(f.match(usercmd))
        self.assertFalse(f.match([envcmd, 'C=ELSE']))
        self.assertFalse(f.match(['env', 'C=xx']))
        self.assertFalse(f.match(['env', 'A=xx']))
        self.assertFalse(f.match(realcmd))
        self.assertFalse(f.match(envcmd))
        self.assertFalse(f.match(envcmd[1:]))
        self.assertEqual(realcmd, f.exec_args(usercmd))
        env = f.get_environment(usercmd)
        self.assertEqual('/some/thing', env.get('A'))
        self.assertEqual('somethingelse', env.get('B'))
        self.assertNotIn('sleep', env.keys())

    def test_EnvFilter_without_leading_env(self):
        envset = ['A=/some/thing', 'B=somethingelse']
        envcmd = ['env'] + envset
        realcmd = ['sleep', '10']
        f = filters.EnvFilter('sleep', 'root', 'A=', 'B=ignored')
        self.assertTrue(f.match(envset + ['sleep']))
        self.assertEqual(realcmd, f.get_command(envcmd + realcmd))
        self.assertEqual(realcmd, f.get_command(envset + realcmd))
        env = f.get_environment(envset + realcmd)
        self.assertEqual('/some/thing', env.get('A'))
        self.assertEqual('somethingelse', env.get('B'))
        self.assertNotIn('sleep', env.keys())

    def test_KillFilter(self):
        if not os.path.exists('/proc/%d' % os.getpid()):
            self.skipTest('Test requires /proc filesystem (procfs)')
        p = subprocess.Popen(['cat'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            f = filters.KillFilter('root', '/bin/cat', '-9', '-HUP')
            f2 = filters.KillFilter('root', '/usr/bin/cat', '-9', '-HUP')
            f3 = filters.KillFilter('root', '/usr/bin/coreutils', '-9', '-HUP')
            usercmd = ['kill', '-ALRM', p.pid]
            self.assertFalse(f.match(usercmd) or f2.match(usercmd))
            usercmd = ['kill', p.pid]
            self.assertFalse(f.match(usercmd) or f2.match(usercmd))
            usercmd = ['kill', '-9', p.pid]
            self.assertTrue(f.match(usercmd) or f2.match(usercmd) or f3.match(usercmd))
            f = filters.KillFilter('root', '/bin/cat')
            f2 = filters.KillFilter('root', '/usr/bin/cat')
            f3 = filters.KillFilter('root', '/usr/bin/coreutils')
            usercmd = ['kill', os.getpid()]
            self.assertFalse(f.match(usercmd) or f2.match(usercmd))
            usercmd = ['kill', 999999]
            self.assertFalse(f.match(usercmd) or f2.match(usercmd))
            usercmd = ['kill', p.pid]
            self.assertTrue(f.match(usercmd) or f2.match(usercmd) or f3.match(usercmd))
            f = filters.KillFilter('root', 'cat')
            f2 = filters.KillFilter('root', 'coreutils')
            usercmd = ['kill', os.getpid()]
            self.assertFalse(f.match(usercmd))
            usercmd = ['kill', p.pid]
            self.assertTrue(f.match(usercmd) or f2.match(usercmd))
            with fixtures.EnvironmentVariable('PATH', '/foo:/bar'):
                self.assertFalse(f.match(usercmd))
            with fixtures.EnvironmentVariable('PATH'):
                self.assertFalse(f.match(usercmd))
        finally:
            p.terminate()
            p.wait()

    def test_KillFilter_no_raise(self):
        """Makes sure ValueError from bug 926412 is gone."""
        f = filters.KillFilter('root', '')
        usercmd = ['notkill', 999999]
        self.assertFalse(f.match(usercmd))
        usercmd = ['kill', 'notapid']
        self.assertFalse(f.match(usercmd))
        self.assertFalse(f.match([]))
        self.assertFalse(f.match(None))

    def test_KillFilter_deleted_exe(self):
        """Makes sure deleted exe's are killed correctly."""
        command = '/bin/commandddddd'
        f = filters.KillFilter('root', command)
        usercmd = ['kill', 1234]
        with mock.patch('os.readlink') as readlink:
            readlink.return_value = command + ' (deleted)'
            with mock.patch('os.path.isfile') as exists:

                def fake_exists(path):
                    return path == command
                exists.side_effect = fake_exists
                self.assertTrue(f.match(usercmd))

    @mock.patch('os.readlink')
    @mock.patch('os.path.isfile')
    def test_KillFilter_upgraded_exe(self, mock_isfile, mock_readlink):
        """Makes sure upgraded exe's are killed correctly."""
        f = filters.KillFilter('root', '/bin/commandddddd')
        command = '/bin/commandddddd'
        usercmd = ['kill', 1234]

        def fake_exists(path):
            return path == command
        mock_readlink.return_value = command + '\x00)90bfb2 (deleted)'
        mock_isfile.side_effect = fake_exists
        self.assertTrue(f.match(usercmd))

    @mock.patch('os.readlink')
    @mock.patch('os.path.isfile')
    @mock.patch('os.path.exists')
    @mock.patch('os.access')
    def test_KillFilter_renamed_exe(self, mock_access, mock_exists, mock_isfile, mock_readlink):
        """Makes sure renamed exe's are killed correctly."""
        command = '/bin/commandddddd'
        f = filters.KillFilter('root', command)
        usercmd = ['kill', 1234]

        def fake_os_func(path, *args):
            return path == command
        mock_readlink.return_value = command + ';90bfb2 (deleted)'
        m = mock.mock_open(read_data=command)
        with mock.patch('builtins.open', m, create=True):
            mock_isfile.side_effect = fake_os_func
            mock_exists.side_effect = fake_os_func
            mock_access.side_effect = fake_os_func
            self.assertTrue(f.match(usercmd))

    def test_ReadFileFilter(self):
        goodfn = '/good/file.name'
        f = filters.ReadFileFilter(goodfn)
        usercmd = ['cat', '/bad/file']
        self.assertFalse(f.match(['cat', '/bad/file']))
        usercmd = ['cat', goodfn]
        self.assertEqual(['/bin/cat', goodfn], f.get_command(usercmd))
        self.assertTrue(f.match(usercmd))

    def test_IpFilter_non_netns(self):
        f = filters.IpFilter(self._ip, 'root')
        self.assertTrue(f.match(['ip', 'link', 'list']))
        self.assertTrue(f.match(['ip', '-s', 'link', 'list']))
        self.assertTrue(f.match(['ip', '-s', '-v', 'netns', 'add']))
        self.assertTrue(f.match(['ip', 'link', 'set', 'interface', 'netns', 'somens']))

    def test_IpFilter_netns(self):
        f = filters.IpFilter(self._ip, 'root')
        self.assertFalse(f.match(['ip', 'netns', 'exec', 'foo']))
        self.assertFalse(f.match(['ip', 'netns', 'exec']))
        self.assertFalse(f.match(['ip', '-s', 'netns', 'exec']))
        self.assertFalse(f.match(['ip', '-l', '42', 'netns', 'exec']))
        self.assertFalse(f.match(['ip', 'net', 'exec', 'foo']))
        self.assertFalse(f.match(['ip', 'netns', 'e', 'foo']))

    def _test_IpFilter_netns_helper(self, action):
        f = filters.IpFilter(self._ip, 'root')
        self.assertTrue(f.match(['ip', 'link', action]))

    def test_IpFilter_netns_add(self):
        self._test_IpFilter_netns_helper('add')

    def test_IpFilter_netns_delete(self):
        self._test_IpFilter_netns_helper('delete')

    def test_IpFilter_netns_list(self):
        self._test_IpFilter_netns_helper('list')

    def test_IpNetnsExecFilter_match(self):
        f = filters.IpNetnsExecFilter(self._ip, 'root')
        self.assertTrue(f.match(['ip', 'netns', 'exec', 'foo', 'ip', 'link', 'list']))
        self.assertTrue(f.match(['ip', 'net', 'exec', 'foo', 'bar']))
        self.assertTrue(f.match(['ip', 'netn', 'e', 'foo', 'bar']))
        self.assertTrue(f.match(['ip', 'net', 'e', 'foo', 'bar']))
        self.assertTrue(f.match(['ip', 'net', 'exe', 'foo', 'bar']))

    def test_IpNetnsExecFilter_nomatch(self):
        f = filters.IpNetnsExecFilter(self._ip, 'root')
        self.assertFalse(f.match(['ip', 'link', 'list']))
        self.assertFalse(f.match(['ip', 'foo', 'bar', 'netns']))
        self.assertFalse(f.match(['ip', '-s', 'netns', 'exec']))
        self.assertFalse(f.match(['ip', '-l', '42', 'netns', 'exec']))
        self.assertFalse(f.match(['ip', 'netns exec', 'foo', 'bar', 'baz']))
        self.assertFalse(f.match([]))
        self.assertFalse(f.match(['ip', 'netns', 'exec']))

    def test_IpNetnsExecFilter_nomatch_nonroot(self):
        f = filters.IpNetnsExecFilter(self._ip, 'user')
        self.assertFalse(f.match(['ip', 'netns', 'exec', 'foo', 'ip', 'link', 'list']))

    def test_match_filter_recurses_exec_command_filter_matches(self):
        filter_list = [filters.IpNetnsExecFilter(self._ip, 'root'), filters.IpFilter(self._ip, 'root')]
        args = ['ip', 'netns', 'exec', 'foo', 'ip', 'link', 'list']
        self.assertIsNotNone(wrapper.match_filter(filter_list, args))

    def test_match_filter_recurses_exec_command_matches_user(self):
        filter_list = [filters.IpNetnsExecFilter(self._ip, 'root'), filters.IpFilter(self._ip, 'user')]
        args = ['ip', 'netns', 'exec', 'foo', 'ip', 'link', 'list']
        self.assertRaises(wrapper.NoFilterMatched, wrapper.match_filter, filter_list, args)

    def test_match_filter_recurses_exec_command_filter_does_not_match(self):
        filter_list = [filters.IpNetnsExecFilter(self._ip, 'root'), filters.IpFilter(self._ip, 'root')]
        args = ['ip', 'netns', 'exec', 'foo', 'ip', 'netns', 'exec', 'bar', 'ip', 'link', 'list']
        self.assertRaises(wrapper.NoFilterMatched, wrapper.match_filter, filter_list, args)

    def test_ChainingRegExpFilter_match(self):
        filter_list = [filters.ChainingRegExpFilter('nice', 'root', 'nice', '-?\\d+'), filters.CommandFilter('cat', 'root')]
        args = ['nice', '5', 'cat', '/a']
        dirs = ['/bin', '/usr/bin']
        self.assertIsNotNone(wrapper.match_filter(filter_list, args, dirs))

    def test_ChainingRegExpFilter_not_match(self):
        filter_list = [filters.ChainingRegExpFilter('nice', 'root', 'nice', '-?\\d+'), filters.CommandFilter('cat', 'root')]
        args_invalid = (['nice', '5', 'ls', '/a'], ['nice', '--5', 'cat', '/a'], ['nice2', '5', 'cat', '/a'], ['nice', 'cat', '/a'], ['nice', '5'])
        dirs = ['/bin', '/usr/bin']
        for args in args_invalid:
            self.assertRaises(wrapper.NoFilterMatched, wrapper.match_filter, filter_list, args, dirs)

    def test_ChainingRegExpFilter_multiple(self):
        filter_list = [filters.ChainingRegExpFilter('ionice', 'root', 'ionice', '-c[0-3]'), filters.ChainingRegExpFilter('ionice', 'root', 'ionice', '-c[0-3]', '-n[0-7]'), filters.CommandFilter('cat', 'root')]
        args = ['ionice', '-c2', '-n7', 'cat', '/a']
        dirs = ['/bin', '/usr/bin']
        self.assertIsNotNone(wrapper.match_filter(filter_list, args, dirs))

    def test_ReadFileFilter_empty_args(self):
        goodfn = '/good/file.name'
        f = filters.ReadFileFilter(goodfn)
        self.assertFalse(f.match([]))
        self.assertFalse(f.match(None))

    def test_exec_dirs_search(self):
        f = filters.CommandFilter('cat', 'root')
        usercmd = ['cat', '/f']
        self.assertTrue(f.match(usercmd))
        self.assertTrue(f.get_command(usercmd, exec_dirs=['/bin', '/usr/bin']) in (['/bin/cat', '/f'], ['/usr/bin/cat', '/f']))

    def test_skips(self):
        usercmd = ['cat', '/']
        filtermatch = wrapper.match_filter(self.filters, usercmd)
        self.assertTrue(filtermatch is self.filters[-1])

    def test_RootwrapConfig(self):
        raw = configparser.RawConfigParser()
        self.assertRaises(configparser.Error, wrapper.RootwrapConfig, raw)
        raw.set('DEFAULT', 'filters_path', '/a,/b')
        config = wrapper.RootwrapConfig(raw)
        self.assertEqual(['/a', '/b'], config.filters_path)
        self.assertEqual(os.environ['PATH'].split(':'), config.exec_dirs)
        with fixtures.EnvironmentVariable('PATH'):
            c = wrapper.RootwrapConfig(raw)
            self.assertEqual([], c.exec_dirs)
        self.assertFalse(config.use_syslog)
        self.assertEqual(logging.handlers.SysLogHandler.LOG_SYSLOG, config.syslog_log_facility)
        self.assertEqual(logging.ERROR, config.syslog_log_level)
        raw.set('DEFAULT', 'exec_dirs', '/a,/x')
        config = wrapper.RootwrapConfig(raw)
        self.assertEqual(['/a', '/x'], config.exec_dirs)
        raw.set('DEFAULT', 'use_syslog', 'oui')
        self.assertRaises(ValueError, wrapper.RootwrapConfig, raw)
        raw.set('DEFAULT', 'use_syslog', 'true')
        config = wrapper.RootwrapConfig(raw)
        self.assertTrue(config.use_syslog)
        raw.set('DEFAULT', 'syslog_log_facility', 'moo')
        self.assertRaises(ValueError, wrapper.RootwrapConfig, raw)
        raw.set('DEFAULT', 'syslog_log_facility', 'local0')
        config = wrapper.RootwrapConfig(raw)
        self.assertEqual(logging.handlers.SysLogHandler.LOG_LOCAL0, config.syslog_log_facility)
        raw.set('DEFAULT', 'syslog_log_facility', 'LOG_AUTH')
        config = wrapper.RootwrapConfig(raw)
        self.assertEqual(logging.handlers.SysLogHandler.LOG_AUTH, config.syslog_log_facility)
        raw.set('DEFAULT', 'syslog_log_level', 'bar')
        self.assertRaises(ValueError, wrapper.RootwrapConfig, raw)
        raw.set('DEFAULT', 'syslog_log_level', 'INFO')
        config = wrapper.RootwrapConfig(raw)
        self.assertEqual(logging.INFO, config.syslog_log_level)

    def test_getlogin(self):
        with mock.patch('os.getlogin') as os_getlogin:
            os_getlogin.return_value = 'foo'
            self.assertEqual('foo', wrapper._getlogin())

    def test_getlogin_bad(self):
        with mock.patch('os.getenv') as os_getenv:
            with mock.patch('os.getlogin') as os_getlogin:
                os_getenv.side_effect = [None, None, 'bar']
                os_getlogin.side_effect = OSError('[Errno 22] Invalid argument')
                self.assertEqual('bar', wrapper._getlogin())
                os_getlogin.assert_called_once_with()
                self.assertEqual(3, os_getenv.call_count)