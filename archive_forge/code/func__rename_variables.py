import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _rename_variables(fstruct, vars, used_vars, new_vars, fs_class, visited):
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for fname, fval in items:
        if isinstance(fval, Variable):
            if fval in new_vars:
                fstruct[fname] = new_vars[fval]
            elif fval in vars:
                new_vars[fval] = _rename_variable(fval, used_vars)
                fstruct[fname] = new_vars[fval]
                used_vars.add(new_vars[fval])
        elif isinstance(fval, fs_class):
            _rename_variables(fval, vars, used_vars, new_vars, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            for var in fval.variables():
                if var in vars and var not in new_vars:
                    new_vars[var] = _rename_variable(var, used_vars)
                    used_vars.add(new_vars[var])
            fstruct[fname] = fval.substitute_bindings(new_vars)
    return fstruct