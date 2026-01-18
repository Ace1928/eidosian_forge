import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
class TestBashCodeGen(tests.TestCase):

    def test_command_names(self):
        data = CompletionData()
        bar = CommandData('bar')
        bar.aliases.append('baz')
        data.commands.append(bar)
        data.commands.append(CommandData('foo'))
        cg = BashCodeGen(data)
        self.assertEqual('bar baz foo', cg.command_names())

    def test_debug_output(self):
        data = CompletionData()
        self.assertEqual('', BashCodeGen(data, debug=False).debug_output())
        self.assertTrue(BashCodeGen(data, debug=True).debug_output())

    def test_brz_version(self):
        data = CompletionData()
        cg = BashCodeGen(data)
        self.assertEqual('%s.' % breezy.version_string, cg.brz_version())
        data.plugins['foo'] = PluginData('foo', '1.0')
        data.plugins['bar'] = PluginData('bar', '2.0')
        cg = BashCodeGen(data)
        self.assertEqual('%s and the following plugins:\n# bar 2.0\n# foo 1.0' % breezy.version_string, cg.brz_version())

    def test_global_options(self):
        data = CompletionData()
        data.global_options.add('--foo')
        data.global_options.add('--bar')
        cg = BashCodeGen(data)
        self.assertEqual('--bar --foo', cg.global_options())

    def test_command_cases(self):
        data = CompletionData()
        bar = CommandData('bar')
        bar.aliases.append('baz')
        bar.options.append(OptionData('--opt'))
        data.commands.append(bar)
        data.commands.append(CommandData('foo'))
        cg = BashCodeGen(data)
        self.assertEqualDiff('\tbar|baz)\n\t\tcmdOpts=( --opt )\n\t\t;;\n\tfoo)\n\t\tcmdOpts=(  )\n\t\t;;\n', cg.command_cases())

    def test_command_case(self):
        cmd = CommandData('cmd')
        cmd.plugin = PluginData('plugger', '1.0')
        bar = OptionData('--bar')
        bar.registry_keys = ['that', 'this']
        bar.error_messages.append('Some error message')
        cmd.options.append(bar)
        cmd.options.append(OptionData('--foo'))
        data = CompletionData()
        data.commands.append(cmd)
        cg = BashCodeGen(data)
        self.assertEqualDiff('\tcmd)\n\t\t# plugin "plugger 1.0"\n\t\t# Some error message\n\t\tcmdOpts=( --bar=that --bar=this --foo )\n\t\tcase $curOpt in\n\t\t\t--bar) optEnums=( that this ) ;;\n\t\tesac\n\t\t;;\n', cg.command_case(cmd))