import asyncio
import logging
import pathlib
import queue
import tempfile
import threading
import wave
from enum import Enum
from typing import (
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
from langchain_core.runnables import RunnableConfig, RunnableSerializable
class RivaAuthMixin(BaseModel):
    """Configuration for the authentication to a Riva service connection."""
    url: Union[AnyHttpUrl, str] = Field(AnyHttpUrl('http://localhost:50051', scheme='http'), description='The full URL where the Riva service can be found.', examples=['http://localhost:50051', 'https://user@pass:riva.example.com'])
    ssl_cert: Optional[str] = Field(None, description="A full path to the file where Riva's public ssl key can be read.")

    @property
    def auth(self) -> 'riva.client.Auth':
        """Return a riva client auth object."""
        riva_client = _import_riva_client()
        url = cast(AnyHttpUrl, self.url)
        use_ssl = url.scheme == 'https'
        url_no_scheme = str(self.url).split('/')[2]
        return riva_client.Auth(ssl_cert=self.ssl_cert, use_ssl=use_ssl, uri=url_no_scheme)

    @validator('url', pre=True, allow_reuse=True)
    @classmethod
    def _validate_url(cls, val: Any) -> AnyHttpUrl:
        """Do some initial conversations for the URL before checking."""
        if isinstance(val, str):
            return cast(AnyHttpUrl, parse_obj_as(AnyHttpUrl, val))
        return cast(AnyHttpUrl, val)