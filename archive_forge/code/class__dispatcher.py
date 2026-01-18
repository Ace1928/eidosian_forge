from .. import exc
from ..sql import sqltypes
class _dispatcher:

    def __init__(self):
        self.specs = {}

    def __call__(self, element, compiler, **kw):
        fn = self.specs.get(compiler.dialect.name, None)
        if not fn:
            try:
                fn = self.specs['default']
            except KeyError as ke:
                raise exc.UnsupportedCompilationError(compiler, type(element), message='%s construct has no default compilation handler.' % type(element)) from ke
        arm = kw.get('add_to_result_map', None)
        if arm:
            arm_collection = []
            kw['add_to_result_map'] = lambda *args: arm_collection.append(args)
        expr = fn(element, compiler, **kw)
        if arm:
            if not arm_collection:
                arm_collection.append((None, None, (element,), sqltypes.NULLTYPE))
            for tup in arm_collection:
                arm(*tup)
        return expr