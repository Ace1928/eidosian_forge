import ast
import re
from collections import OrderedDict
def _create_subscript_in(atom, root):
    """Find / create and insert object defined by `atom` from list `root`

    The `atom` has an index, defined in ``atom.obj_id``.  If `root` is long
    enough to contain this index, return the object at that index.  Otherwise,
    extend `root` with None elements to contain index ``atom.obj_id``, then
    create a new object via ``atom.obj_type()``, insert at the end of the list,
    and return this object.

    Can therefore modify `root` in place.
    """
    curr_n = len(root)
    index = atom.obj_id
    if curr_n > index:
        return root[index]
    obj = atom.obj_type()
    root += [None] * (index - curr_n) + [obj]
    return obj