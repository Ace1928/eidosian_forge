import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
class SingletonMeta(type):
    """
    Metaclass that creates a Singleton instance.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            else:
                cls._instances[cls]._set_session(*args, **kwargs)
            return cls._instances[cls]