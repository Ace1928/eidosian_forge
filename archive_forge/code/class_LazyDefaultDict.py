from collections import defaultdict
from functools import lru_cache
import boto3
from boto3.exceptions import ResourceNotExistsError
from boto3.resources.base import ServiceResource
from botocore.client import BaseClient
from botocore.config import Config
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import BOTO_MAX_RETRIES
class LazyDefaultDict(defaultdict):
    """
    LazyDefaultDict(default_factory[, ...]) --> dict with default factory

    The default factory is call with the key argument to produce
    a new value when a key is not present, in __getitem__ only.
    A LazyDefaultDict compares equal to a dict with the same items.
    All remaining arguments are treated the same as if they were
    passed to the dict constructor, including keyword arguments.
    """

    def __missing__(self, key):
        """
        __missing__(key) # Called by __getitem__ for missing key; pseudo-code:
          if self.default_factory is None: raise KeyError((key,))
          self[key] = value = self.default_factory(key)
          return value
        """
        self[key] = self.default_factory(key)
        return self[key]