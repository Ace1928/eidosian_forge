import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
def StandardDecorator():
    """
    A decorator to standardize and log the execution of methods within the neural network classes.
    This decorator aims to provide insights into method calls, arguments passed, and the time taken for execution.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f'Executing {func.__name__} with args: {args} and kwargs: {kwargs}')
            result = func(*args, **kwargs)
            logging.info(f'Executed {func.__name__} successfully.')
            return result
        return wrapper
    return decorator