import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import logging, parse_datetime
def _populate_from_raw(self) -> None:
    """Populate fields from raw dictionary.

        Called in __post_init__ + each time the Inference Endpoint is updated.
        """
    self.name = self.raw['name']
    self.repository = self.raw['model']['repository']
    self.status = self.raw['status']['state']
    self.url = self.raw['status'].get('url')
    self.framework = self.raw['model']['framework']
    self.revision = self.raw['model']['revision']
    self.task = self.raw['model']['task']
    self.created_at = parse_datetime(self.raw['status']['createdAt'])
    self.updated_at = parse_datetime(self.raw['status']['updatedAt'])
    self.type = self.raw['type']