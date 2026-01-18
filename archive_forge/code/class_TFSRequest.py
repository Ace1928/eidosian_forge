import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
@lazyclass
@dataclass
class TFSRequest:
    data: Any

    def to_data(self):
        if isinstance(self.data, str):
            return {'inputs': [self.data]}
        if isinstance(self.data, list):
            return {'inputs': self.data}
        if isinstance(self.data, dict):
            if self.data.get('inputs'):
                return self.data
            if self.data.get('text'):
                if isinstance(self.data['text'], str):
                    return {'inputs': [self.data['text']]}
                return {'inputs': self.data['text']}
        return {'inputs': self.data}