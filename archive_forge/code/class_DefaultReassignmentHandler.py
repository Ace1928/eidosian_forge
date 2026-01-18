from abc import ABC, abstractmethod
from typing import Set, Optional, Awaitable
from google.cloud.pubsublite.types import Partition
class DefaultReassignmentHandler(ReassignmentHandler):

    def handle_reassignment(self, before: Set[Partition], after: Set[Partition]) -> Optional[Awaitable]:
        return None