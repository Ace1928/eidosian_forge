import base64
import hashlib
import hmac
import json
import logging
import queue
import threading
from datetime import datetime
from queue import Queue
from time import mktime
from typing import Any, Dict, Generator, Iterator, List, Mapping, Optional, Type
from urllib.parse import urlencode, urlparse, urlunparse
from wsgiref.handlers import format_date_time
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import (
class ChatSparkLLM(BaseChatModel):
    """iFlyTek Spark large language model.

    To use, you should pass `app_id`, `api_key`, `api_secret`
    as a named parameter to the constructor OR set environment
    variables ``IFLYTEK_SPARK_APP_ID``, ``IFLYTEK_SPARK_API_KEY`` and
    ``IFLYTEK_SPARK_API_SECRET``

    Example:
        .. code-block:: python

        client = ChatSparkLLM(
            spark_app_id="<app_id>",
            spark_api_key="<api_key>",
            spark_api_secret="<api_secret>"
        )

    Extra infos:
        1. Get app_id, api_key, api_secret from the iFlyTek Open Platform Console:
            https://console.xfyun.cn/services/bm35
        2. By default, iFlyTek Spark LLM V3.0 is invoked.
            If you need to invoke other versions, please configure the corresponding
            parameters(spark_api_url and spark_llm_domain) according to the document:
            https://www.xfyun.cn/doc/spark/Web.html
        3. It is necessary to ensure that the app_id used has a license for
            the corresponding model version.
        4. If you encounter problems during use, try getting help at:
            https://console.xfyun.cn/workorder/commit
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'spark_app_id': 'IFLYTEK_SPARK_APP_ID', 'spark_api_key': 'IFLYTEK_SPARK_API_KEY', 'spark_api_secret': 'IFLYTEK_SPARK_API_SECRET', 'spark_api_url': 'IFLYTEK_SPARK_API_URL', 'spark_llm_domain': 'IFLYTEK_SPARK_LLM_DOMAIN'}
    client: Any = None
    spark_app_id: Optional[str] = None
    spark_api_key: Optional[str] = None
    spark_api_secret: Optional[str] = None
    spark_api_url: Optional[str] = None
    spark_llm_domain: Optional[str] = None
    spark_user_id: str = 'lc_user'
    streaming: bool = False
    request_timeout: int = Field(30, alias='timeout')
    temperature: float = 0.5
    top_k: int = 4
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get('model_kwargs', {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f'Found {field_name} supplied twice.')
            if field_name not in all_required_field_names:
                logger.warning(f'WARNING! {field_name} is not default parameter.\n                    {field_name} was transferred to model_kwargs.\n                    Please confirm that {field_name} is what you intended.')
                extra[field_name] = values.pop(field_name)
        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(f'Parameters {invalid_model_kwargs} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.')
        values['model_kwargs'] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values['spark_app_id'] = get_from_dict_or_env(values, 'spark_app_id', 'IFLYTEK_SPARK_APP_ID')
        values['spark_api_key'] = get_from_dict_or_env(values, 'spark_api_key', 'IFLYTEK_SPARK_API_KEY')
        values['spark_api_secret'] = get_from_dict_or_env(values, 'spark_api_secret', 'IFLYTEK_SPARK_API_SECRET')
        values['spark_api_url'] = get_from_dict_or_env(values, 'spark_api_url', 'IFLYTEK_SPARK_API_URL', 'wss://spark-api.xf-yun.com/v3.1/chat')
        values['spark_llm_domain'] = get_from_dict_or_env(values, 'spark_llm_domain', 'IFLYTEK_SPARK_LLM_DOMAIN', 'generalv3')
        values['model_kwargs']['temperature'] = values['temperature'] or cls.temperature
        values['model_kwargs']['top_k'] = values['top_k'] or cls.top_k
        values['client'] = _SparkLLMClient(app_id=values['spark_app_id'], api_key=values['spark_api_key'], api_secret=values['spark_api_secret'], api_url=values['spark_api_url'], spark_domain=values['spark_llm_domain'], model_kwargs=values['model_kwargs'])
        return values

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk
        self.client.arun([_convert_message_to_dict(m) for m in messages], self.spark_user_id, self.model_kwargs, self.streaming)
        for content in self.client.subscribe(timeout=self.request_timeout):
            if 'data' not in content:
                continue
            delta = content['data']
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(str(chunk.content), chunk=cg_chunk)
            yield cg_chunk

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(messages=messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)
        self.client.arun([_convert_message_to_dict(m) for m in messages], self.spark_user_id, self.model_kwargs, False)
        completion = {}
        llm_output = {}
        for content in self.client.subscribe(timeout=self.request_timeout):
            if 'usage' in content:
                llm_output['token_usage'] = content['usage']
            if 'data' not in content:
                continue
            completion = content['data']
        message = _convert_dict_to_message(completion)
        generations = [ChatGeneration(message=message)]
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return 'spark-llm-chat'