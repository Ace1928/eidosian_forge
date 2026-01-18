from __future__ import annotations
import gc
import niquests
from urllib.parse import urljoin
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
class CaptureEvent(BaseEvent):
    """
    The Capture Event
    """
    event: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    distinct_id: Optional[str] = None
    timestamp: Optional[int] = None
    token: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(None)

    @property
    def is_valid(self) -> bool:
        """
        Returns True if the event is valid
        """
        return self.event and self.distinct_id

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