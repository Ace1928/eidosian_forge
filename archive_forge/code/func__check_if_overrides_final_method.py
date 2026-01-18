from abc import ABCMeta
@staticmethod
def _check_if_overrides_final_method(name, bases):
    for base in bases:
        base_class_method = getattr(base, name, False)
        if getattr(base_class_method, '__final__', False):
            raise TypeError(f'Method {name} is finalized in {base}, it cannot be overridden')