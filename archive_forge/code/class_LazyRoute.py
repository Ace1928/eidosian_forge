import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
@lazyclass
@dataclass
class LazyRoute:
    path: str
    method: Optional[str] = 'POST'
    data_key: Optional[str] = 'inputs'
    params_key: Optional[str] = 'params'
    params: Optional[Dict[str, Any]] = None
    prefix_payload: Optional[str] = None
    is_async: Optional[bool] = False
    decode_json: Optional[bool] = True

    def get_config(self, base_url, data=None, **config):
        p = self.params.copy() if self.params else {}
        if config:
            for k, v in config.items():
                if (self.params and k in self.params or not self.params) and v:
                    p[k] = v
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        if not self.path.startswith('/'):
            self.path = '/' + self.path
        pc = {self.params_key: p} if self.params_key else p
        if data:
            pc[self.data_key] = data
        if self.prefix_payload:
            pc = {self.prefix_payload: pc}
        return {'method': self.method, 'url': base_url + self.path, self.pkey: pc, 'decode_json': self.decode_json}

    @property
    def pkey(self):
        if self.method in ['POST']:
            return 'json'
        if self.method in ['GET']:
            return 'params'
        return 'data'