import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
@lazyclass
@dataclass
class LazyAPIModel:
    url: str
    header: LazyHeader
    routes: Dict[str, LazyRoute]
    session: Optional[LazySession] = None

    @property
    def sess(self):
        if not self.session:
            self.session = LazySession(header=self.header.config)
        return self.session

    async def async_call(self, route, *args, **kwargs):
        return await self.sess.async_fetch(**self.routes[route].get_config(*args, base_url=self.url, **kwargs))

    def get(self, route, *args, **kwargs):
        if route not in self.routes:
            raise ValueError
        if self.routes[route].is_async and kwargs.get('call_async', False):
            return self.async_call(route, *args, **kwargs)
        return self.sess(**self.routes[route].get_config(*args, base_url=self.url, **kwargs))

    def __call__(self, route, *args, **kwargs):
        return self.get(route, *args, **kwargs)