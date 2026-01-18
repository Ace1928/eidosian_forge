import os
import weakref
import torch
def _get_python_class(qualified_name):
    return _name_to_pyclass.get(qualified_name, None)