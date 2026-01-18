import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _dict_to_tuple_key(dictionary):
    """Converts a dictionary to a tuple that can be used as an immutable key.

    The resulting key is always sorted so that logically equivalent
    dictionaries always produce an identical tuple for a key.

    Args:
        dictionary: the dictionary to use as the key.

    Returns:
        A tuple representing the dictionary in it's naturally sorted ordering.
    """
    return tuple(sorted(dictionary.items()))