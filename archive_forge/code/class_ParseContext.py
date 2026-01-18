from __future__ import print_function
from __future__ import unicode_literals
import contextlib
from cmakelang import common
class ParseContext(object):
    """Global context passed through every function in the parse stack."""

    def __init__(self, parse_db=None, lint_ctx=None, config=None):
        if parse_db is None:
            from cmakelang.parse.funs import get_parse_db
            parse_db = get_parse_db()
        self.parse_db = parse_db
        if lint_ctx is None:
            lint_ctx = MockEverything()
        self.lint_ctx = lint_ctx
        if config is None:
            from cmakelang import configuration
            config = configuration.Configuration()
        self.config = config
        self.argstack = []

    @contextlib.contextmanager
    def pusharg(self, node):
        self.argstack.append(node)
        yield None
        if not self.argstack:
            raise common.InternalError('Unexpected empty argstack, expected {}'.format(node))
        if self.argstack[-1] is not node:
            raise common.InternalError('Unexpected node {} on argstack, expecting {}'.format(self.argstack[-1], node))
        self.argstack.pop(-1)