from ...commands import Command, plugin_cmds, register_command
from .. import TestCaseWithMemoryTransport
class TestCommandEncoding(TestCaseWithMemoryTransport):

    def test_exact(self):

        def bzr(*args, **kwargs):
            kwargs['encoding'] = 'ascii'
            return self.run_bzr_raw(*args, **kwargs)[0]
        register_command(cmd_echo_exact)
        try:
            self.assertEqual(b'foo', bzr('echo-exact foo'))
            self.assertRaises(UnicodeEncodeError, bzr, ['echo-exact', 'fooµ'])
        finally:
            plugin_cmds.remove('echo-exact')

    def test_strict_utf8(self):

        def bzr(*args, **kwargs):
            kwargs['encoding'] = 'utf-8'
            return self.run_bzr_raw(*args, **kwargs)[0]
        register_command(cmd_echo_strict)
        try:
            self.assertEqual(b'foo', bzr('echo-strict foo'))
            expected = 'fooµ'
            expected = expected.encode('utf-8')
            self.assertEqual(expected, bzr(['echo-strict', 'fooµ']))
        finally:
            plugin_cmds.remove('echo-strict')

    def test_strict_ascii(self):

        def bzr(*args, **kwargs):
            kwargs['encoding'] = 'ascii'
            return self.run_bzr_raw(*args, **kwargs)[0]
        register_command(cmd_echo_strict)
        try:
            self.assertEqual(b'foo', bzr('echo-strict foo'))
            self.assertRaises(UnicodeEncodeError, bzr, ['echo-strict', 'fooµ'])
        finally:
            plugin_cmds.remove('echo-strict')

    def test_replace_utf8(self):

        def bzr(*args, **kwargs):
            kwargs['encoding'] = 'utf-8'
            return self.run_bzr_raw(*args, **kwargs)[0]
        register_command(cmd_echo_replace)
        try:
            self.assertEqual(b'foo', bzr('echo-replace foo'))
            self.assertEqual('fooµ'.encode(), bzr(['echo-replace', 'fooµ']))
        finally:
            plugin_cmds.remove('echo-replace')

    def test_replace_ascii(self):

        def bzr(*args, **kwargs):
            kwargs['encoding'] = 'ascii'
            return self.run_bzr_raw(*args, **kwargs)[0]
        register_command(cmd_echo_replace)
        try:
            self.assertEqual(b'foo', bzr('echo-replace foo'))
            self.assertEqual(b'foo?', bzr(['echo-replace', 'fooµ']))
        finally:
            plugin_cmds.remove('echo-replace')