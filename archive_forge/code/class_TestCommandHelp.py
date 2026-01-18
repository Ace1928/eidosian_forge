import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class TestCommandHelp(tests.TestCase):
    """Tests for help on commands."""

    def assertCmdHelp(self, expected, cmd):
        self.assertEqualDiff(textwrap.dedent(expected), cmd.get_help_text())

    def test_command_help_includes_see_also(self):

        class cmd_WithSeeAlso(commands.Command):
            __doc__ = 'A sample command.'
            _see_also = ['foo', 'bar']
        self.assertCmdHelp('Purpose: A sample command.\nUsage:   brz WithSeeAlso\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nSee also: bar, foo\n', cmd_WithSeeAlso())

    def test_get_help_text(self):
        """Commands have a get_help_text method which returns their help."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.'
        self.assertCmdHelp('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\n', cmd_Demo())
        cmd = cmd_Demo()
        helptext = cmd.get_help_text()
        self.assertStartsWith(helptext, 'Purpose: A sample command.\nUsage:   brz Demo')
        self.assertEndsWith(helptext, '  -v, --verbose  Display more information.\n\n')

    def test_command_with_additional_see_also(self):

        class cmd_WithSeeAlso(commands.Command):
            __doc__ = 'A sample command.'
            _see_also = ['foo', 'bar']
        cmd = cmd_WithSeeAlso()
        helptext = cmd.get_help_text(['gam'])
        self.assertEndsWith(helptext, '  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nSee also: bar, foo, gam\n')

    def test_command_only_additional_see_also(self):

        class cmd_WithSeeAlso(commands.Command):
            __doc__ = 'A sample command.'
        cmd = cmd_WithSeeAlso()
        helptext = cmd.get_help_text(['gam'])
        self.assertEndsWith(helptext, '  -v, --verbose  Display more information.\n\nSee also: gam\n')

    def test_get_help_topic(self):
        """The help topic for a Command is its name()."""

        class cmd_foo_bar(commands.Command):
            __doc__ = 'A sample command.'
        cmd = cmd_foo_bar()
        self.assertEqual(cmd.name(), cmd.get_help_topic())

    def test_formatted_help_text(self):
        """Help text should be plain text by default."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.\n\n            :Examples:\n                Example 1::\n\n                    cmd arg1\n\n                Example 2::\n\n                    cmd arg2\n\n                A code block follows.\n\n                ::\n\n                    brz Demo something\n            '
        cmd = cmd_Demo()
        helptext = cmd.get_help_text()
        self.assertEqualDiff('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nExamples:\n    Example 1:\n\n        cmd arg1\n\n    Example 2:\n\n        cmd arg2\n\n    A code block follows.\n\n        brz Demo something\n\n', helptext)
        helptext = cmd.get_help_text(plain=False)
        self.assertEqualDiff(':Purpose: A sample command.\n:Usage:   brz Demo\n\n:Options:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\n:Examples:\n    Example 1::\n\n        cmd arg1\n\n    Example 2::\n\n        cmd arg2\n\n    A code block follows.\n\n    ::\n\n        brz Demo something\n\n', helptext)

    def test_concise_help_text(self):
        """Concise help text excludes the descriptive sections."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.\n\n            Blah blah blah.\n\n            :Examples:\n                Example 1::\n\n                    cmd arg1\n            '
        cmd = cmd_Demo()
        helptext = cmd.get_help_text()
        self.assertEqualDiff('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nDescription:\n  Blah blah blah.\n\nExamples:\n    Example 1:\n\n        cmd arg1\n\n', helptext)
        helptext = cmd.get_help_text(verbose=False)
        self.assertEqualDiff('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nSee brz help Demo for more details and examples.\n\n', helptext)

    def test_help_custom_section_ordering(self):
        """Custom descriptive sections should remain in the order given."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.\n\nBlah blah blah.\n\n:Formats:\n  Interesting stuff about formats.\n\n:Examples:\n  Example 1::\n\n    cmd arg1\n\n:Tips:\n  Clever things to keep in mind.\n'
        cmd = cmd_Demo()
        helptext = cmd.get_help_text()
        self.assertEqualDiff('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nDescription:\n  Blah blah blah.\n\nFormats:\n  Interesting stuff about formats.\n\nExamples:\n  Example 1:\n\n    cmd arg1\n\nTips:\n  Clever things to keep in mind.\n\n', helptext)

    def test_help_text_custom_usage(self):
        """Help text may contain a custom usage section."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.\n\n            :Usage:\n                cmd Demo [opts] args\n\n                cmd Demo -h\n\n            Blah blah blah.\n            '
        cmd = cmd_Demo()
        helptext = cmd.get_help_text()
        self.assertEqualDiff('Purpose: A sample command.\nUsage:\n    cmd Demo [opts] args\n\n    cmd Demo -h\n\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nDescription:\n  Blah blah blah.\n\n', helptext)