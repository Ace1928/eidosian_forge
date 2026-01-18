import sys
import xml.parsers.expat
def _apply_defusing(defused_mod):
    assert defused_mod is sys.modules[defused_mod.__name__]
    stdlib_name = defused_mod.__origin__
    __import__(stdlib_name, {}, {}, ['*'])
    stdlib_mod = sys.modules[stdlib_name]
    stdlib_names = set(dir(stdlib_mod))
    for name, obj in vars(defused_mod).items():
        if name.startswith('_') or name not in stdlib_names:
            continue
        setattr(stdlib_mod, name, obj)
    return stdlib_mod