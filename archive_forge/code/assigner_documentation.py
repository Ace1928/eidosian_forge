from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager, Set
from google.cloud.pubsublite.types.partition import Partition

    An assigner will deliver a continuous stream of assignments when called into. Perform all necessary work with the
    assignment before attempting to get the next one.
    