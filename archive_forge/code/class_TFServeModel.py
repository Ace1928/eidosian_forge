import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
class TFServeModel:

    def __init__(self, url: str, configs: List[TFSModelConfig], ver: str='v1', **kwargs):
        self.url = url
        self.configs = configs
        self.ver = ver
        self.headers = {'Content-Type': 'application/json'}
        if kwargs:
            self.headers.update(kwargs.get('headers'))
        self.sess = requests.Session()
        self.sess.headers.update(self.headers)
        self.endpoints = {config.name: TFSModelEndpoint(url, config=config, ver=ver, sess=self.sess, headers=self.headers) for config in configs}
        self.default_model = kwargs.get('default_model') or configs[0].name
        self.available_models = [config.name for config in configs]
        self.validate_models()

    def predict(self, data, model: str=None, label: str=None, ver: Union[str, int]=None, **kwargs):
        if model:
            assert model in self.available_models
        model = model or self.default_model
        return self.endpoints[model].predict(data, label=label, ver=ver, **kwargs)

    async def aio_predict(self, data, model: str=None, label: str=None, ver: Union[str, int]=None, **kwargs):
        if model:
            assert model in self.available_models
        model = model or self.default_model
        return await self.endpoints[model].aio_predict(data, label=label, ver=ver, **kwargs)

    @property
    def api_status(self):
        return {model: self.endpoints[model].is_alive for model in self.available_models}

    @timed_cache(seconds=600)
    def get_api_metadata(self):
        return {model: self.endpoints[model].get_metadata() for model in self.available_models}

    @async_cache
    async def get_aio_api_metadata(self):
        return {model: await self.endpoints[model].get_aio_metadata() for model in self.available_models}

    @classmethod
    def from_config_file(cls, url, filepath: str, ver: str='v1', **kwargs):
        configs = TFSConfig.from_config_file(filepath, as_obj=True)
        return TFServeModel(url=url, configs=configs, ver=ver, **kwargs)

    def validate_models(self):
        for model in list(self.available_models):
            if not self.endpoints[model].is_alive:
                self.available_models.remove(model)
                _ = self.endpoints.pop(model)
        if self.default_model not in self.available_models:
            self.default_model = self.available_models[0]