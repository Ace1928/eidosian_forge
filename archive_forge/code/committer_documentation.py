from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager
from google.cloud.pubsublite_v1 import Cursor

        Flushes pending commits and waits for all outstanding commit responses from the server.

        Raises:
          GoogleAPICallError: When the committer terminates in failure.
        