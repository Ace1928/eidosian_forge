import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
class TestOptionMisc(TestCase):

    def test_is_hidden(self):
        registry = controldir.ControlDirFormatRegistry()
        bzr.register_metadir(registry, 'hidden', 'HiddenFormat', 'hidden help text', hidden=True)
        bzr.register_metadir(registry, 'visible', 'VisibleFormat', 'visible help text', hidden=False)
        format = option.RegistryOption('format', '', registry, str)
        self.assertTrue(format.is_hidden('hidden'))
        self.assertFalse(format.is_hidden('visible'))

    def test_short_name(self):
        registry = controldir.ControlDirFormatRegistry()
        opt = option.RegistryOption('format', help='', registry=registry)
        self.assertEqual(None, opt.short_name())
        opt = option.RegistryOption('format', short_name='F', help='', registry=registry)
        self.assertEqual('F', opt.short_name())

    def test_option_custom_help(self):
        the_opt = option.Option.OPTIONS['help']
        orig_help = the_opt.help[:]
        my_opt = option.custom_help('help', 'suggest lottery numbers')
        self.assertEqual('suggest lottery numbers', my_opt.help)
        self.assertEqual(orig_help, the_opt.help)

    def test_short_value_switches(self):
        reg = registry.Registry()
        reg.register('short', 'ShortChoice')
        reg.register('long', 'LongChoice')
        ropt = option.RegistryOption('choice', '', reg, value_switches=True, short_value_switches={'short': 's'})
        opts, args = parse([ropt], ['--short'])
        self.assertEqual('ShortChoice', opts.choice)
        opts, args = parse([ropt], ['-s'])
        self.assertEqual('ShortChoice', opts.choice)