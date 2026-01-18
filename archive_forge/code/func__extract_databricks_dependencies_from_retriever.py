import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set
def _extract_databricks_dependencies_from_retriever(retriever, dependency_dict: DefaultDict[str, List[Any]]):
    try:
        from langchain.embeddings import DatabricksEmbeddings as LegacyDatabricksEmbeddings
        from langchain.vectorstores import DatabricksVectorSearch as LegacyDatabricksVectorSearch
    except ImportError:
        from langchain_community.embeddings import DatabricksEmbeddings as LegacyDatabricksEmbeddings
        from langchain_community.vectorstores import DatabricksVectorSearch as LegacyDatabricksVectorSearch
    from langchain_community.embeddings import DatabricksEmbeddings
    from langchain_community.vectorstores import DatabricksVectorSearch
    vectorstore = getattr(retriever, 'vectorstore', None)
    if vectorstore:
        if isinstance(vectorstore, (DatabricksVectorSearch, LegacyDatabricksVectorSearch)):
            index = vectorstore.index
            dependency_dict[_DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY].append(index.name)
            dependency_dict[_DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY].append(index.endpoint_name)
        embeddings = getattr(vectorstore, 'embeddings', None)
        if isinstance(embeddings, (DatabricksEmbeddings, LegacyDatabricksEmbeddings)):
            dependency_dict[_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY].append(embeddings.endpoint)
        elif callable(getattr(vectorstore, '_is_databricks_managed_embeddings', None)) and vectorstore._is_databricks_managed_embeddings():
            dependency_dict[_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY].append('_is_databricks_managed_embeddings')