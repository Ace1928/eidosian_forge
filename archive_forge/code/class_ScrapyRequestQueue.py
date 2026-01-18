import marshal
import pickle
from os import PathLike
from pathlib import Path
from typing import Union
from queuelib import queue
from scrapy.utils.request import request_from_dict
class ScrapyRequestQueue(queue_class):

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        return cls()

    def peek(self):
        """Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            """
        try:
            s = super().peek()
        except AttributeError as ex:
            raise NotImplementedError("The underlying queue class does not implement 'peek'") from ex
        return s