from __future__ import annotations
import gc
import niquests
from urllib.parse import urljoin
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
class IdentifyEvent(BaseEvent):
    """
    The Identify Event
    """
    distinct_id: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    token: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """
        Returns True if the event is valid
        """
        return self.distinct_id

    def prepare_request(self, exclude_none: Optional[bool]=True, batched: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
        """
        Formats the event for batching
        """
        data = self.model_dump(exclude_none=exclude_none, **kwargs)
        if batched:
            data['properties']['distinct_id'] = data.pop('distinct_id')
            if 'timestamp' in data:
                data['properties']['timestamp'] = data.pop('timestamp')
        return data