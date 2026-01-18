import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
import numpy as np
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
@staticmethod
def ExactRetrievalStrategy() -> 'ExactRetrievalStrategy':
    """Used to perform brute force / exact
        nearest neighbor search via script_score."""
    return ExactRetrievalStrategy()