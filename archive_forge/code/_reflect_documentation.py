import sys
from jsonschema.compat import PY3

    Retrieve a Python object by its fully qualified name from the global Python
    module namespace.  The first part of the name, that describes a module,
    will be discovered and imported.  Each subsequent part of the name is
    treated as the name of an attribute of the object specified by all of the
    name which came before it.  For example, the fully-qualified name of this
    object is 'twisted.python.reflect.namedAny'.

    @type name: L{str}
    @param name: The name of the object to return.

    @raise InvalidName: If the name is an empty string, starts or ends with
        a '.', or is otherwise syntactically incorrect.

    @raise ModuleNotFound: If the name is syntactically correct but the
        module it specifies cannot be imported because it does not appear to
        exist.

    @raise ObjectNotFound: If the name is syntactically correct, includes at
        least one '.', but the module it specifies cannot be imported because
        it does not appear to exist.

    @raise AttributeError: If an attribute of an object along the way cannot be
        accessed, or a module along the way is not found.

    @return: the Python object identified by 'name'.
    