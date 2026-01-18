import ast
import re
from collections import OrderedDict
def _create_obj_in(atom, root):
    """Find / create object defined in `atom` in dict-like given by `root`

    Returns corresponding value if there is already a key matching
    `atom.obj_id` in `root`.

    Otherwise, create new object with ``atom.obj_type`, insert into dictionary,
    and return new object.

    Can therefore modify `root` in place.
    """
    name = atom.obj_id
    obj = root.get(name, NoValue)
    if obj is not NoValue:
        return obj
    obj = atom.obj_type()
    root[name] = obj
    return obj