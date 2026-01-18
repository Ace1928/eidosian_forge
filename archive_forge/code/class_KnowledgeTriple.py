from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
class KnowledgeTriple(NamedTuple):
    """Knowledge triple in the graph."""
    subject: str
    predicate: str
    object_: str

    @classmethod
    def from_string(cls, triple_string: str) -> 'KnowledgeTriple':
        """Create a KnowledgeTriple from a string."""
        subject, predicate, object_ = triple_string.strip().split(', ')
        subject = subject[1:]
        object_ = object_[:-1]
        return cls(subject, predicate, object_)