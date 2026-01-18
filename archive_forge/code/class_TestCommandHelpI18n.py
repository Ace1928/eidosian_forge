import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class TestCommandHelpI18n(tests.TestCase):
    """Tests for help on translated commands."""

    def setUp(self):
        super().setUp()
        self.overrideAttr(i18n, '_translations', ZzzTranslationsForDoc())

    def assertCmdHelp(self, expected, cmd):
        self.assertEqualDiff(textwrap.dedent(expected), cmd.get_help_text())

    def test_command_help_includes_see_also(self):

        class cmd_WithSeeAlso(commands.Command):
            __doc__ = 'A sample command.'
            _see_also = ['foo', 'bar']
        self.assertCmdHelp('zz{{:Purpose: zz{{A sample command.}}\n}}zz{{:Usage:   brz WithSeeAlso\n}}\nzz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nzz{{:See also: bar, foo}}\n', cmd_WithSeeAlso())

    def test_get_help_text(self):
        """Commands have a get_help_text method which returns their help."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.'
        self.assertCmdHelp('zz{{:Purpose: zz{{A sample command.}}\n}}zz{{:Usage:   brz Demo\n}}\nzz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\n', cmd_Demo())

    def test_command_with_additional_see_also(self):

        class cmd_WithSeeAlso(commands.Command):
            __doc__ = 'A sample command.'
            _see_also = ['foo', 'bar']
        cmd = cmd_WithSeeAlso()
        helptext = cmd.get_help_text(['gam'])
        self.assertEndsWith(helptext, '  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nzz{{:See also: bar, foo, gam}}\n')

    def test_command_only_additional_see_also(self):

        class cmd_WithSeeAlso(commands.Command):
            __doc__ = 'A sample command.'
        cmd = cmd_WithSeeAlso()
        helptext = cmd.get_help_text(['gam'])
        self.assertEndsWith(helptext, 'zz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nzz{{:See also: gam}}\n')

    def test_help_custom_section_ordering(self):
        """Custom descriptive sections should remain in the order given."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.\n\n            Blah blah blah.\n\n            :Formats:\n              Interesting stuff about formats.\n\n            :Examples:\n              Example 1::\n\n                cmd arg1\n\n            :Tips:\n              Clever things to keep in mind.\n            '
        self.assertCmdHelp('zz{{:Purpose: zz{{A sample command.}}\n}}zz{{:Usage:   brz Demo\n}}\nzz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nDescription:\n  zz{{zz{{Blah blah blah.}}\n\n}}:Formats:\n  zz{{Interesting stuff about formats.}}\n\nExamples:\n  zz{{Example 1::}}\n\n    zz{{cmd arg1}}\n\nTips:\n  zz{{Clever things to keep in mind.}}\n\n', cmd_Demo())

    def test_help_text_custom_usage(self):
        """Help text may contain a custom usage section."""

        class cmd_Demo(commands.Command):
            __doc__ = 'A sample command.\n\n            :Usage:\n                cmd Demo [opts] args\n\n                cmd Demo -h\n\n            Blah blah blah.\n            '
        self.assertCmdHelp('zz{{:Purpose: zz{{A sample command.}}\n}}zz{{:Usage:\n    zz{{cmd Demo [opts] args}}\n\n    zz{{cmd Demo -h}}\n\n}}\nzz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nDescription:\n  zz{{zz{{Blah blah blah.}}\n\n}}\n', cmd_Demo())