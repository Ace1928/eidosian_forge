from __future__ import annotations
from pydantic import model_validator
from lazyops.types import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
class SlackPayload(BaseModel):
    token: Optional[str] = None
    team_id: Optional[str] = None
    team_domain: Optional[str] = None
    enterprise_id: Optional[str] = None
    enterprise_name: Optional[str] = None
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    command: Optional[str] = None
    text: Optional[str] = None
    response_url: Optional[str] = None
    trigger_id: Optional[str] = None
    api_app_id: Optional[str] = None
    ccommand: Optional[str] = None
    ctext: Optional[str] = None

    @model_validator(mode='after')
    def set_mutable_context(self):
        """
        Sets the ctext
        """
        if self.text is not None:
            self.ctext = self.text
        if self.command is not None:
            self.ccommand = self.command
        return self