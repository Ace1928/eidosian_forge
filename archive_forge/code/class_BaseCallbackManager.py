from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class BaseCallbackManager(CallbackManagerMixin):
    """Base callback manager that handles callbacks from LangChain."""

    def __init__(self, handlers: List[BaseCallbackHandler], inheritable_handlers: Optional[List[BaseCallbackHandler]]=None, parent_run_id: Optional[UUID]=None, *, tags: Optional[List[str]]=None, inheritable_tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, inheritable_metadata: Optional[Dict[str, Any]]=None) -> None:
        """Initialize callback manager."""
        self.handlers: List[BaseCallbackHandler] = handlers
        self.inheritable_handlers: List[BaseCallbackHandler] = inheritable_handlers or []
        self.parent_run_id: Optional[UUID] = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    def copy(self: T) -> T:
        """Copy the callback manager."""
        return self.__class__(handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata)

    @property
    def is_async(self) -> bool:
        """Whether the callback manager is async."""
        return False

    def add_handler(self, handler: BaseCallbackHandler, inherit: bool=True) -> None:
        """Add a handler to the callback manager."""
        if handler not in self.handlers:
            self.handlers.append(handler)
        if inherit and handler not in self.inheritable_handlers:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)
        self.inheritable_handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler], inherit: bool=True) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = []
        self.inheritable_handlers = []
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def set_handler(self, handler: BaseCallbackHandler, inherit: bool=True) -> None:
        """Set handler as the only handler on the callback manager."""
        self.set_handlers([handler], inherit=inherit)

    def add_tags(self, tags: List[str], inherit: bool=True) -> None:
        for tag in tags:
            if tag in self.tags:
                self.remove_tags([tag])
        self.tags.extend(tags)
        if inherit:
            self.inheritable_tags.extend(tags)

    def remove_tags(self, tags: List[str]) -> None:
        for tag in tags:
            self.tags.remove(tag)
            self.inheritable_tags.remove(tag)

    def add_metadata(self, metadata: Dict[str, Any], inherit: bool=True) -> None:
        self.metadata.update(metadata)
        if inherit:
            self.inheritable_metadata.update(metadata)

    def remove_metadata(self, keys: List[str]) -> None:
        for key in keys:
            self.metadata.pop(key)
            self.inheritable_metadata.pop(key)