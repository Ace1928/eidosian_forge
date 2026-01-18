import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set
def _extract_databricks_dependencies_from_chat_model(chat_model, dependency_dict: DefaultDict[str, List[Any]]):
    try:
        from langchain.chat_models import ChatDatabricks as LegacyChatDatabricks
    except ImportError:
        from langchain_community.chat_models import ChatDatabricks as LegacyChatDatabricks
    from langchain_community.chat_models import ChatDatabricks
    if isinstance(chat_model, (LegacyChatDatabricks, ChatDatabricks)):
        dependency_dict[_DATABRICKS_CHAT_ENDPOINT_NAME_KEY].append(chat_model.endpoint)