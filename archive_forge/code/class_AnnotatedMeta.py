import abc
import collections
import collections.abc
import operator
import sys
import typing
class AnnotatedMeta(typing.GenericMeta):
    """Metaclass for Annotated"""

    def __new__(cls, name, bases, namespace, **kwargs):
        if any((b is not object for b in bases)):
            raise TypeError('Cannot subclass ' + str(Annotated))
        return super().__new__(cls, name, bases, namespace, **kwargs)

    @property
    def __metadata__(self):
        return self._subs_tree()[2]

    def _tree_repr(self, tree):
        cls, origin, metadata = tree
        if not isinstance(origin, tuple):
            tp_repr = typing._type_repr(origin)
        else:
            tp_repr = origin[0]._tree_repr(origin)
        metadata_reprs = ', '.join((repr(arg) for arg in metadata))
        return f'{cls}[{tp_repr}, {metadata_reprs}]'

    def _subs_tree(self, tvars=None, args=None):
        if self is Annotated:
            return Annotated
        res = super()._subs_tree(tvars=tvars, args=args)
        if isinstance(res[1], tuple) and res[1][0] is Annotated:
            sub_tp = res[1][1]
            sub_annot = res[1][2]
            return (Annotated, sub_tp, sub_annot + res[2])
        return res

    def _get_cons(self):
        """Return the class used to create instance of this type."""
        if self.__origin__ is None:
            raise TypeError('Cannot get the underlying type of a non-specialized Annotated type.')
        tree = self._subs_tree()
        while isinstance(tree, tuple) and tree[0] is Annotated:
            tree = tree[1]
        if isinstance(tree, tuple):
            return tree[0]
        else:
            return tree

    @typing._tp_cache
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        if self.__origin__ is not None:
            return super().__getitem__(params)
        elif not isinstance(params, tuple) or len(params) < 2:
            raise TypeError('Annotated[...] should be instantiated with at least two arguments (a type and an annotation).')
        else:
            msg = 'Annotated[t, ...]: t must be a type.'
            tp = typing._type_check(params[0], msg)
            metadata = tuple(params[1:])
        return self.__class__(self.__name__, self.__bases__, _no_slots_copy(self.__dict__), tvars=_type_vars((tp,)), args=(tp, metadata), origin=self)

    def __call__(self, *args, **kwargs):
        cons = self._get_cons()
        result = cons(*args, **kwargs)
        try:
            result.__orig_class__ = self
        except AttributeError:
            pass
        return result

    def __getattr__(self, attr):
        if self.__origin__ is not None and (not _is_dunder(attr)):
            return getattr(self._get_cons(), attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if _is_dunder(attr) or attr.startswith('_abc_'):
            super().__setattr__(attr, value)
        elif self.__origin__ is None:
            raise AttributeError(attr)
        else:
            setattr(self._get_cons(), attr, value)

    def __instancecheck__(self, obj):
        raise TypeError('Annotated cannot be used with isinstance().')

    def __subclasscheck__(self, cls):
        raise TypeError('Annotated cannot be used with issubclass().')