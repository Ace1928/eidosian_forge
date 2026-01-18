from pythran.analyses import OptimizableComprehension
from pythran.passmanager import Transformation
from pythran.transformations.normalize_tuples import ConvertToTuple
from pythran.conversion import mangle
from pythran.utils import attr_to_path, path_to_attr
import gast as ast

    Transforms list comprehension into intrinsics.
    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(y) : return (x for x in y)")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(ComprehensionPatterns, node)
    >>> 'map' in pm.dump(backend.Python, node)
    True

    >>> node = ast.parse("def foo(y) : return [0 for _ in builtins.range(y)]")
    >>> _, node = pm.apply(ComprehensionPatterns, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(y):
        return ([0] * builtins.len(builtins.range(y)))
    