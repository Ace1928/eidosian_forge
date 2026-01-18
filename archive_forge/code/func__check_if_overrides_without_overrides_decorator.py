from abc import ABCMeta
@staticmethod
def _check_if_overrides_without_overrides_decorator(name, value, bases):
    is_override = getattr(value, '__override__', False)
    for base in bases:
        base_class_method = getattr(base, name, False)
        if not base_class_method or not callable(base_class_method) or getattr(base_class_method, '__ignored__', False):
            continue
        if not is_override:
            raise TypeError(f'Method {name} overrides method from {base} but does not have @override decorator')