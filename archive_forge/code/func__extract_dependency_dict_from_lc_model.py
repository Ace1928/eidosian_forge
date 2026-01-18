import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set
def _extract_dependency_dict_from_lc_model(lc_model, dependency_dict: DefaultDict[str, List[Any]]):
    """
    This function contains the logic to examine a non-Runnable component of a langchain model.
    The logic here does not cover all legacy chains. If you need to support a custom chain,
    you need to monkey patch this function.
    """
    if lc_model is None:
        return
    _extract_databricks_dependencies_from_chat_model(lc_model, dependency_dict)
    _extract_databricks_dependencies_from_retriever(lc_model, dependency_dict)
    _extract_databricks_dependencies_from_llm(lc_model, dependency_dict)
    for attr_name in _LEGACY_MODEL_ATTR_SET:
        _extract_dependency_dict_from_lc_model(getattr(lc_model, attr_name, None), dependency_dict)