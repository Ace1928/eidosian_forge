from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler

        Adds the provied key and value to the cache. If the key already exists, no
        change is made. Returns itself for method chaining.
        