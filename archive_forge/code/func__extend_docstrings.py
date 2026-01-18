import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def _extend_docstrings(cls):
    if cls.__doc__ and cls.__doc__.startswith('+'):
        cls.__doc__ = _append_doc(cls.__bases__[0].__doc__, cls.__doc__)
    for name, attr in cls.__dict__.items():
        if attr.__doc__ and attr.__doc__.startswith('+'):
            for c in (c for base in cls.__bases__ for c in base.mro()):
                doc = getattr(getattr(c, name), '__doc__')
                if doc:
                    attr.__doc__ = _append_doc(doc, attr.__doc__)
                    break
    return cls