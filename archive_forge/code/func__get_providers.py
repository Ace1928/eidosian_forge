from __future__ import annotations
import json
import urllib.request
import uuid
from typing import Callable
from urllib.parse import quote
def _get_providers(provider):
    if isinstance(provider, TileProvider):
        flat[provider.name] = provider
    else:
        for prov in provider.values():
            _get_providers(prov)