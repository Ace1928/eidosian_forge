import io
from ...migration import MigrationContext
from ...testing import assert_raises
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import is_false
from ...testing import is_not_
from ...testing import is_true
from ...testing import ne_
from ...testing.fixtures import TestBase
def _assert_impl_steps(self, *steps):
    to_check = self.context.output_buffer.getvalue()
    self.context.impl.output_buffer = buf = io.StringIO()
    for step in steps:
        if step == 'BEGIN':
            self.context.impl.emit_begin()
        elif step == 'COMMIT':
            self.context.impl.emit_commit()
        else:
            self.context.impl._exec(step)
    eq_(to_check, buf.getvalue())