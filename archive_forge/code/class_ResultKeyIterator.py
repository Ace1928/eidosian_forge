import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
class ResultKeyIterator:
    """Iterates over the results of paginated responses.

    Each iterator is associated with a single result key.
    Iterating over this object will give you each element in
    the result key list.

    :param pages_iterator: An iterator that will give you
        pages of results (a ``PageIterator`` class).
    :param result_key: The JMESPath expression representing
        the result key.

    """

    def __init__(self, pages_iterator, result_key):
        self._pages_iterator = pages_iterator
        self.result_key = result_key

    def __iter__(self):
        for page in self._pages_iterator:
            results = self.result_key.search(page)
            if results is None:
                results = []
            yield from results