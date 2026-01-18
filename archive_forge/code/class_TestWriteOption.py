import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class TestWriteOption(tests.TestCase):
    """Tests for writing texts extracted from options in pot format"""

    def pot_from_option(self, opt, context=None, note='test'):
        sio = StringIO()
        exporter = export_pot._PotExporter(sio)
        if context is None:
            context = export_pot._ModuleContext('nowhere', 0)
        export_pot._write_option(exporter, context, opt, note)
        return sio.getvalue()

    def test_option_without_help(self):
        opt = option.Option('helpless')
        self.assertEqual('', self.pot_from_option(opt))

    def test_option_with_help(self):
        opt = option.Option('helpful', help='Info.')
        self.assertContainsString(self.pot_from_option(opt), '\n# help of \'helpful\' test\nmsgid "Info."\n')

    def test_option_hidden(self):
        opt = option.Option('hidden', help='Unseen.', hidden=True)
        self.assertEqual('', self.pot_from_option(opt))

    def test_option_context_missing(self):
        context = export_pot._ModuleContext('remote.py', 3)
        opt = option.Option('metaphor', help='Not a literal in the source.')
        self.assertContainsString(self.pot_from_option(opt, context), "#: remote.py:3\n# help of 'metaphor' test\n")

    def test_option_context_string(self):
        s = 'Literally.'
        context = export_pot._ModuleContext('local.py', 3, ({}, {s: 17}))
        opt = option.Option('example', help=s)
        self.assertContainsString(self.pot_from_option(opt, context), "#: local.py:17\n# help of 'example' test\n")

    def test_registry_option_title(self):
        opt = option.RegistryOption.from_kwargs('group', help='Pick one.', title='Choose!')
        pot = self.pot_from_option(opt)
        self.assertContainsString(pot, '\n# title of \'group\' test\nmsgid "Choose!"\n')
        self.assertContainsString(pot, '\n# help of \'group\' test\nmsgid "Pick one."\n')

    def test_registry_option_title_context_missing(self):
        context = export_pot._ModuleContext('theory.py', 3)
        opt = option.RegistryOption.from_kwargs('abstract', title='Unfounded!')
        self.assertContainsString(self.pot_from_option(opt, context), "#: theory.py:3\n# title of 'abstract' test\n")

    def test_registry_option_title_context_string(self):
        s = 'Grounded!'
        context = export_pot._ModuleContext('practice.py', 3, ({}, {s: 144}))
        opt = option.RegistryOption.from_kwargs('concrete', title=s)
        self.assertContainsString(self.pot_from_option(opt, context), "#: practice.py:144\n# title of 'concrete' test\n")

    def test_registry_option_value_switches(self):
        opt = option.RegistryOption.from_kwargs('switch', help='Flip one.', value_switches=True, enum_switch=False, red='Big.', green='Small.')
        pot = self.pot_from_option(opt)
        self.assertContainsString(pot, '\n# help of \'switch\' test\nmsgid "Flip one."\n')
        self.assertContainsString(pot, '\n# help of \'switch=red\' test\nmsgid "Big."\n')
        self.assertContainsString(pot, '\n# help of \'switch=green\' test\nmsgid "Small."\n')

    def test_registry_option_value_switches_hidden(self):
        reg = registry.Registry()

        class Hider:
            hidden = True
        reg.register('new', 1, 'Current.')
        reg.register('old', 0, 'Legacy.', info=Hider())
        opt = option.RegistryOption('protocol', 'Talking.', reg, value_switches=True, enum_switch=False)
        pot = self.pot_from_option(opt)
        self.assertContainsString(pot, '\n# help of \'protocol\' test\nmsgid "Talking."\n')
        self.assertContainsString(pot, '\n# help of \'protocol=new\' test\nmsgid "Current."\n')
        self.assertNotContainsString(pot, "'protocol=old'")