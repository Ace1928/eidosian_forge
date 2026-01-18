import collections
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class UnstructuredBaseLoader(BaseLoader, ABC):
    """Base Loader that uses `Unstructured`."""

    def __init__(self, mode: str='single', post_processors: Optional[List[Callable]]=None, **unstructured_kwargs: Any):
        """Initialize with file path."""
        try:
            import unstructured
        except ImportError:
            raise ValueError('unstructured package not found, please install it with `pip install unstructured`')
        _valid_modes = {'single', 'elements', 'paged'}
        if mode not in _valid_modes:
            raise ValueError(f'Got {mode} for `mode`, but should be one of `{_valid_modes}`')
        self.mode = mode
        if not satisfies_min_unstructured_version('0.5.4'):
            if 'strategy' in unstructured_kwargs:
                unstructured_kwargs.pop('strategy')
        self.unstructured_kwargs = unstructured_kwargs
        self.post_processors = post_processors or []

    @abstractmethod
    def _get_elements(self) -> List:
        """Get elements."""

    @abstractmethod
    def _get_metadata(self) -> dict:
        """Get metadata."""

    def _post_process_elements(self, elements: list) -> list:
        """Applies post processing functions to extracted unstructured elements.
        Post processing functions are str -> str callables are passed
        in using the post_processors kwarg when the loader is instantiated."""
        for element in elements:
            for post_processor in self.post_processors:
                element.apply(post_processor)
        return elements

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        elements = self._get_elements()
        self._post_process_elements(elements)
        if self.mode == 'elements':
            for element in elements:
                metadata = self._get_metadata()
                if hasattr(element, 'metadata'):
                    metadata.update(element.metadata.to_dict())
                if hasattr(element, 'category'):
                    metadata['category'] = element.category
                yield Document(page_content=str(element), metadata=metadata)
        elif self.mode == 'paged':
            text_dict: Dict[int, str] = {}
            meta_dict: Dict[int, Dict] = {}
            for idx, element in enumerate(elements):
                metadata = self._get_metadata()
                if hasattr(element, 'metadata'):
                    metadata.update(element.metadata.to_dict())
                page_number = metadata.get('page_number', 1)
                if page_number not in text_dict:
                    text_dict[page_number] = str(element) + '\n\n'
                    meta_dict[page_number] = metadata
                else:
                    text_dict[page_number] += str(element) + '\n\n'
                    meta_dict[page_number].update(metadata)
            for key in text_dict.keys():
                yield Document(page_content=text_dict[key], metadata=meta_dict[key])
        elif self.mode == 'single':
            metadata = self._get_metadata()
            text = '\n\n'.join([str(el) for el in elements])
            yield Document(page_content=text, metadata=metadata)
        else:
            raise ValueError(f'mode of {self.mode} not supported.')