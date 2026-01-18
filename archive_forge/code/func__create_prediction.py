from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _create_prediction(self, prompt: str, **kwargs: Any) -> Prediction:
    try:
        import replicate as replicate_python
    except ImportError:
        raise ImportError('Could not import replicate python package. Please install it with `pip install replicate`.')
    if self.version_obj is None:
        if ':' in self.model:
            model_str, version_str = self.model.split(':')
            model = replicate_python.models.get(model_str)
            self.version_obj = model.versions.get(version_str)
        else:
            model = replicate_python.models.get(self.model)
            self.version_obj = model.latest_version
    if self.prompt_key is None:
        input_properties = sorted(self.version_obj.openapi_schema['components']['schemas']['Input']['properties'].items(), key=lambda item: item[1].get('x-order', 0))
        self.prompt_key = input_properties[0][0]
    input_: Dict = {self.prompt_key: prompt, **self.model_kwargs, **kwargs}
    if ':' not in self.model:
        return replicate_python.models.predictions.create(self.model, input=input_)
    else:
        return replicate_python.predictions.create(version=self.version_obj, input=input_)