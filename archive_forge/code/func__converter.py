import builtins
import json
import numbers
import operator
def _converter(value, classinfo):
    """Convert value (str) to number, otherwise return None if is not possible"""
    for one_info in classinfo:
        if issubclass(one_info, numbers.Number):
            try:
                return one_info(value)
            except ValueError:
                pass