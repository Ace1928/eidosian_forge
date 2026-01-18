import time as _time
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta, timezone, tzinfo
class BasicTweetHandler(metaclass=ABCMeta):
    """
    Minimal implementation of `TweetHandler`.

    Counts the number of Tweets and decides when the client should stop
    fetching them.
    """

    def __init__(self, limit=20):
        self.limit = limit
        self.counter = 0
        '\n        A flag to indicate to the client whether to stop fetching data given\n        some condition (e.g., reaching a date limit).\n        '
        self.do_stop = False
        '\n        Stores the id of the last fetched Tweet to handle pagination.\n        '
        self.max_id = None

    def do_continue(self):
        """
        Returns `False` if the client should stop fetching Tweets.
        """
        return self.counter < self.limit and (not self.do_stop)