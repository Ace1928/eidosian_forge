from __future__ import annotations
import builtins
from . import Image, _imagingmath

    Evaluates an image expression.

    :param expression: A string containing a Python-style expression.
    :param options: Values to add to the evaluation context.  You
                    can either use a dictionary, or one or more keyword
                    arguments.
    :return: The evaluated expression. This is usually an image object, but can
             also be an integer, a floating point value, or a pixel tuple,
             depending on the expression.
    