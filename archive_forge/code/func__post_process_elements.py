import collections
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _post_process_elements(self, elements: list) -> list:
    """Applies post processing functions to extracted unstructured elements.
        Post processing functions are str -> str callables are passed
        in using the post_processors kwarg when the loader is instantiated."""
    for element in elements:
        for post_processor in self.post_processors:
            element.apply(post_processor)
    return elements