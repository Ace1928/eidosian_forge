from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import exceptions as cache_exceptions
Wraps a function and caches its result.