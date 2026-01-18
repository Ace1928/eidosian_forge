from abc import ABC, abstractmethod
from typing import AsyncIterable, Tuple
from fastapi import HTTPException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.schemas import chat, completions, embeddings
class ProviderAdapter(ABC):
    """ """

    @classmethod
    @abstractmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def model_to_completions(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def completions_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def chat_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def check_keys_against_mapping(cls, mapping, payload):
        for k1, k2 in mapping.items():
            if k2 in payload:
                raise HTTPException(status_code=400, detail=f'Invalid parameter {k2}. Use {k1} instead.')