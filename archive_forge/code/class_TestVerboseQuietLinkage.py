import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
class TestVerboseQuietLinkage(TestCase):

    def check(self, parser, level, args):
        option._verbosity_level = 0
        opts, args = parser.parse_args(args)
        self.assertEqual(level, option._verbosity_level)

    def test_verbose_quiet_linkage(self):
        parser = option.get_optparser([v for k, v in sorted(option.Option.STD_OPTIONS.items())])
        self.check(parser, 0, [])
        self.check(parser, 1, ['-v'])
        self.check(parser, 2, ['-v', '-v'])
        self.check(parser, -1, ['-q'])
        self.check(parser, -2, ['-qq'])
        self.check(parser, -1, ['-v', '-v', '-q'])
        self.check(parser, 2, ['-q', '-v', '-v'])
        self.check(parser, 0, ['--no-verbose'])
        self.check(parser, 0, ['-v', '-q', '--no-quiet'])