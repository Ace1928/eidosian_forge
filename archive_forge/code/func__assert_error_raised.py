from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _assert_error_raised(func, error, failure_comment):

    def inner_func(*args, **kwargs):
        error_raised = False
        try:
            func(*args, **kwargs)
        except error:
            error_raised = True
        assert error_raised, failure_comment
    return inner_func