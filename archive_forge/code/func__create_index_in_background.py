from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def _create_index_in_background(self):
    if self._have_index or self._creating_index:
        return
    self._creating_index = True
    self._logger.debug('Trying to create a vector index.')
    thread = Thread(target=self._create_index, daemon=True)
    thread.start()