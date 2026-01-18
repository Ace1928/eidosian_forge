import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def _get_typeddict_qualifiers(annotation_type):
    while True:
        annotation_origin = get_origin(annotation_type)
        if annotation_origin is Annotated:
            annotation_args = get_args(annotation_type)
            if annotation_args:
                annotation_type = annotation_args[0]
            else:
                break
        elif annotation_origin is Required:
            yield Required
            annotation_type, = get_args(annotation_type)
        elif annotation_origin is NotRequired:
            yield NotRequired
            annotation_type, = get_args(annotation_type)
        elif annotation_origin is ReadOnly:
            yield ReadOnly
            annotation_type, = get_args(annotation_type)
        else:
            break