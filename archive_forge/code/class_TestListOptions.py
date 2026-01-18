import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
class TestListOptions(TestCase):
    """Tests for ListOption, used to specify lists on the command-line."""

    def parse(self, options, args):
        parser = option.get_optparser(options)
        return parser.parse_args(args)

    def test_list_option(self):
        options = [option.ListOption('hello', type=str)]
        opts, args = self.parse(options, ['--hello=world', '--hello=sailor'])
        self.assertEqual(['world', 'sailor'], opts.hello)

    def test_list_option_with_dash(self):
        options = [option.ListOption('with-dash', type=str)]
        opts, args = self.parse(options, ['--with-dash=world', '--with-dash=sailor'])
        self.assertEqual(['world', 'sailor'], opts.with_dash)

    def test_list_option_no_arguments(self):
        options = [option.ListOption('hello', type=str)]
        opts, args = self.parse(options, [])
        self.assertEqual([], opts.hello)

    def test_list_option_with_int_type(self):
        options = [option.ListOption('hello', type=int)]
        opts, args = self.parse(options, ['--hello=2', '--hello=3'])
        self.assertEqual([2, 3], opts.hello)

    def test_list_option_with_int_type_can_be_reset(self):
        options = [option.ListOption('hello', type=int)]
        opts, args = self.parse(options, ['--hello=2', '--hello=3', '--hello=-', '--hello=5'])
        self.assertEqual([5], opts.hello)

    def test_list_option_can_be_reset(self):
        """Passing an option of '-' to a list option should reset the list."""
        options = [option.ListOption('hello', type=str)]
        opts, args = self.parse(options, ['--hello=a', '--hello=b', '--hello=-', '--hello=c'])
        self.assertEqual(['c'], opts.hello)

    def test_option_callback_list(self):
        """Test callbacks work for list options."""
        cb_calls = []

        def cb(option, name, value, parser):
            cb_calls.append((option, name, value[:], parser))
        options = [option.ListOption('hello', type=str, custom_callback=cb)]
        opts, args = self.parse(options, ['--hello=world', '--hello=mars', '--hello=-'])
        self.assertEqual(3, len(cb_calls))
        opt, name, value, parser = cb_calls[0]
        self.assertEqual('hello', name)
        self.assertEqual(['world'], value)
        opt, name, value, parser = cb_calls[1]
        self.assertEqual('hello', name)
        self.assertEqual(['world', 'mars'], value)
        opt, name, value, parser = cb_calls[2]
        self.assertEqual('hello', name)
        self.assertEqual([], value)

    def test_list_option_param_name(self):
        """Test list options can have their param_name set."""
        options = [option.ListOption('hello', type=str, param_name='greeting')]
        opts, args = self.parse(options, ['--hello=world', '--hello=sailor'])
        self.assertEqual(['world', 'sailor'], opts.greeting)