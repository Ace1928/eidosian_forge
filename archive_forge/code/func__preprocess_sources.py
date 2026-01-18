from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
def _preprocess_sources(self, sources: list) -> list:
    """Checks if the provided sources are string paths. If they are, convert
        to NeuralDB document objects.

        Args:
            sources: list of either string paths to PDF, DOCX or CSV files, or
            NeuralDB document objects.
        """
    from thirdai import neural_db as ndb
    if not sources:
        return sources
    preprocessed_sources = []
    for doc in sources:
        if not isinstance(doc, str):
            preprocessed_sources.append(doc)
        elif doc.lower().endswith('.pdf'):
            preprocessed_sources.append(ndb.PDF(doc))
        elif doc.lower().endswith('.docx'):
            preprocessed_sources.append(ndb.DOCX(doc))
        elif doc.lower().endswith('.csv'):
            preprocessed_sources.append(ndb.CSV(doc))
        else:
            raise RuntimeError(f'Could not automatically load {doc}. Only files with .pdf, .docx, or .csv extensions can be loaded automatically. For other formats, please use the appropriate document object from the ThirdAI library.')
    return preprocessed_sources