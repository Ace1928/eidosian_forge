from collections import deque
from . import errors, revision
Get the children for a key

        Returns a list containg the children keys. A KeyError will be raised
        if the key is not in the graph.

        :param keys: Key to check (eg revision_id)
        :return: A list of children
        