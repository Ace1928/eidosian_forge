from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Union
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra
from requests import Response
class GenericRequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library."""
    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    auth: Optional[Any] = None
    response_content_type: Literal['text', 'json'] = 'text'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def requests(self) -> Requests:
        return Requests(headers=self.headers, aiosession=self.aiosession, auth=self.auth)

    def _get_resp_content(self, response: Response) -> Union[str, Dict[str, Any]]:
        if self.response_content_type == 'text':
            return response.text
        elif self.response_content_type == 'json':
            return response.json()
        else:
            raise ValueError(f'Invalid return type: {self.response_content_type}')

    async def _aget_resp_content(self, response: aiohttp.ClientResponse) -> Union[str, Dict[str, Any]]:
        if self.response_content_type == 'text':
            return await response.text()
        elif self.response_content_type == 'json':
            return await response.json()
        else:
            raise ValueError(f'Invalid return type: {self.response_content_type}')

    def get(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """GET the URL and return the text."""
        return self._get_resp_content(self.requests.get(url, **kwargs))

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """POST to the URL and return the text."""
        return self._get_resp_content(self.requests.post(url, data, **kwargs))

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """PATCH the URL and return the text."""
        return self._get_resp_content(self.requests.patch(url, data, **kwargs))

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """PUT the URL and return the text."""
        return self._get_resp_content(self.requests.put(url, data, **kwargs))

    def delete(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """DELETE the URL and return the text."""
        return self._get_resp_content(self.requests.delete(url, **kwargs))

    async def aget(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """GET the URL and return the text asynchronously."""
        async with self.requests.aget(url, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def apost(self, url: str, data: Dict[str, Any], **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """POST to the URL and return the text asynchronously."""
        async with self.requests.apost(url, data, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def apatch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """PATCH the URL and return the text asynchronously."""
        async with self.requests.apatch(url, data, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def aput(self, url: str, data: Dict[str, Any], **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """PUT the URL and return the text asynchronously."""
        async with self.requests.aput(url, data, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def adelete(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """DELETE the URL and return the text asynchronously."""
        async with self.requests.adelete(url, **kwargs) as response:
            return await self._aget_resp_content(response)