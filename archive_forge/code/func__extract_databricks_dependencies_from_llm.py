import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set
def _extract_databricks_dependencies_from_llm(llm, dependency_dict: DefaultDict[str, List[Any]]):
    try:
        from langchain.llms import Databricks as LegacyDatabricks
    except ImportError:
        from langchain_community.llms import Databricks as LegacyDatabricks
    from langchain_community.llms import Databricks
    if isinstance(llm, (LegacyDatabricks, Databricks)):
        dependency_dict[_DATABRICKS_LLM_ENDPOINT_NAME_KEY].append(llm.endpoint_name)