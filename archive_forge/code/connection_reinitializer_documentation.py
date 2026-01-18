from typing import Generic
from abc import ABCMeta, abstractmethod
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsublite.internal.wire.connection import (
Reinitialize a connection. Must ensure no calls to the associated RetryingConnection
        occur until this completes.

        Args:
            connection: The connection to reinitialize

        Raises:
            GoogleAPICallError: If it fails to reinitialize.
        