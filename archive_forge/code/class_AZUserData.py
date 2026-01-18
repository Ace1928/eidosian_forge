from __future__ import annotations
import time
from .base import BaseModel
from pydantic import Field, model_validator
from typing import Dict, Optional, List, Any
class AZUserData(UserDataBase):
    """
    User Data
    """
    user_id: str
    multifactor: Optional[List[str]] = None
    expiration_ts: Optional[int] = None

    @model_validator(mode='after')
    def set_expiration_ts(self):
        """
        Sets the Expiration Timestamp
        """
        if self.expiration_ts is None:
            self.expiration_ts = int(time.time()) + (self.settings.user_data_expiration or 60)
        return self

    @property
    def is_expired(self) -> bool:
        """
        Returns True if the User Data is Expired
        """
        return self.expiration_ts < int(time.time())