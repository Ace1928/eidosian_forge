import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
@lazyclass
@dataclass
class LazyAPIConfig:
    url: str
    user: Optional[str] = None
    key: Optional[str] = None
    token: Optional[str] = None
    default_params: Optional[Dict[str, str]] = None
    params_key: Optional[str] = None
    data_key: Optional[str] = None
    default_fetch: Optional[str] = None
    default_async: Optional[str] = None
    route_config: Optional[Dict[str, LazyRoute]] = None