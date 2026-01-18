from abc import ABC, abstractmethod
from typing import Dict, List, Union
from langchain_core.documents import Document
class AddableMixin(ABC):
    """Mixin class that supports adding texts."""

    @abstractmethod
    def add(self, texts: Dict[str, Document]) -> None:
        """Add more documents."""