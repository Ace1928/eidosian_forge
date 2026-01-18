import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError
def _get_type_description(annotation):
    """
    Given an annotation, return the (type, description) for the parameter.
    If you provide an annotation that is somehow both a string and a callable,
    the behavior is undefined.
    """
    if annotation is _empty:
        return (None, None)
    elif callable(annotation):
        return (annotation, None)
    elif isinstance(annotation, str):
        return (None, annotation)
    elif isinstance(annotation, tuple):
        try:
            arg1, arg2 = annotation
        except ValueError as e:
            raise AnnotationError(annotation) from e
        else:
            if callable(arg1) and isinstance(arg2, str):
                return (arg1, arg2)
            elif isinstance(arg1, str) and callable(arg2):
                return (arg2, arg1)
    raise AnnotationError(annotation)