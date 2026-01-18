import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
class ArgSignatureTest(fixtures.TestBase):
    """test that all visit_XYZ() in :class:`_sql.Compiler` subclasses have
    ``**kw``, for #8988.

    This test uses runtime code inspection.   Does not need to be a
    ``__backend__`` test as it only needs to run once provided all target
    dialects have been imported.

    For third party dialects, the suite would be run with that third
    party as a "--dburi", which means its compiler classes will have been
    imported by the time this test runs.

    """

    def _all_subclasses():
        for d in dialects.__all__:
            if not d.startswith('_'):
                importlib.import_module('sqlalchemy.dialects.%s' % d)
        stack = [Compiled]
        while stack:
            cls = stack.pop(0)
            stack.extend(cls.__subclasses__())
            yield cls

    @testing.fixture(params=list(_all_subclasses()))
    def all_subclasses(self, request):
        yield request.param

    def test_all_visit_methods_accept_kw(self, all_subclasses):
        cls = all_subclasses
        for k in cls.__dict__:
            if k.startswith('visit_'):
                meth = getattr(cls, k)
                insp = inspect_getfullargspec(meth)
                is_not_none(insp.varkw, f'Compiler visit method {cls.__name__}.{k}() does not accommodate for **kw in its argument signature')