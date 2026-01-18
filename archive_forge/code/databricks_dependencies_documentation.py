import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

    Detects the databricks dependencies of a langchain model and returns a dictionary of
    detected endpoint names and index names.

    lc_model can be an arbitrary [chain that is built with LCEL](https://python.langchain.com
    /docs/modules/chains#lcel-chains), which is a langchain_core.runnables.RunnableSerializable.
    [Legacy chains](https://python.langchain.com/docs/modules/chains#legacy-chains) have limited
    support. Only RetrievalQA, StuffDocumentsChain, ReduceDocumentsChain, RefineDocumentsChain,
    MapRerankDocumentsChain, MapReduceDocumentsChain, BaseConversationalRetrievalChain are
    supported. If you need to support a custom chain, you need to monkey patch
    the function mlflow.langchain.databricks_dependencies._extract_dependency_dict_from_lc_model().

    For an LCEL chain, all the langchain_core.runnables.RunnableSerializable nodes will be
    traversed.

    If a retriever is found, it will be used to extract the databricks vector search and embeddings
    dependencies.
    If an llm is found, it will be used to extract the databricks llm dependencies.
    If a chat_model is found, it will be used to extract the databricks chat dependencies.
    