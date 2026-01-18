from typing import Any, Callable, Dict, Type
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.language_models.llms import BaseLLM
def _import_weight_only_quantization() -> Any:
    from langchain_community.llms.weight_only_quantization import WeightOnlyQuantPipeline
    return WeightOnlyQuantPipeline