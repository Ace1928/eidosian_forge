import collections.abc
import io
import itertools
import types
import typing
def _is_normal_subtype(left: NormalizedType, right: NormalizedType, forward_refs: typing.Optional[typing.Mapping[str, type]]) -> typing.Optional[bool]:
    if isinstance(left.origin, ForwardRef):
        left = normalize(eval_forward_ref(left.origin, forward_refs=forward_refs))
    if isinstance(right.origin, ForwardRef):
        right = normalize(eval_forward_ref(right.origin, forward_refs=forward_refs))
    if right.origin is typing.Any:
        return True
    if is_union(right.origin) and is_union(left.origin):
        return _is_origin_subtype_args(left.args, right.args, forward_refs)
    if is_union(right.origin):
        return optional_any((_is_normal_subtype(left, a, forward_refs) for a in right.args))
    if is_union(left.origin):
        return optional_all((_is_normal_subtype(a, right, forward_refs) for a in left.args))
    if right.origin is Literal:
        if left.origin is not Literal:
            return False
        return set(left.args).issubset(set(right.args))
    if isinstance(left.origin, typing.TypeVar) and isinstance(right.origin, typing.TypeVar):
        if left.origin is right.origin:
            return True
        left_bound = getattr(left.origin, '__bound__', None)
        right_bound = getattr(right.origin, '__bound__', None)
        if right_bound is None or left_bound is None:
            return unknown
        return _is_normal_subtype(normalize(left_bound), normalize(right_bound), forward_refs)
    if isinstance(right.origin, typing.TypeVar):
        return unknown
    if isinstance(left.origin, typing.TypeVar):
        left_bound = getattr(left.origin, '__bound__', None)
        if left_bound is None:
            return unknown
        return _is_normal_subtype(normalize(left_bound), right, forward_refs)
    if not left.args and (not right.args):
        return _is_origin_subtype(left.origin, right.origin)
    if not right.args:
        return _is_origin_subtype(left.origin, right.origin)
    if _is_origin_subtype(left.origin, right.origin):
        return _is_origin_subtype_args(left.args, right.args, forward_refs)
    return False