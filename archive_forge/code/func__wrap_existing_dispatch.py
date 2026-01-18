from .. import exc
from ..sql import sqltypes
def _wrap_existing_dispatch(element, compiler, **kw):
    try:
        return existing_dispatch(element, compiler, **kw)
    except exc.UnsupportedCompilationError as uce:
        raise exc.UnsupportedCompilationError(compiler, type(element), message='%s construct has no default compilation handler.' % type(element)) from uce