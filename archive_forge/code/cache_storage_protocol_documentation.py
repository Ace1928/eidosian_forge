from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol
Checks if the context is valid for the storage manager.
        This method should not return anything, but log message or raise an exception
        if the context is invalid.

        In case of raising an exception, we not handle it and let the exception to be
        propagated.

        check_context is called only once at the moment of creating `@st.cache_data`
        decorator for specific function, so it is not called for every cache hit.

        Parameters
        ----------
        context: CacheStorageContext
            The context to check for the storage manager, dummy function_key in context
            will be used, since it is not computed at the point of calling this method.

        Raises
        ------
        InvalidCacheStorageContext
            Raised if the cache storage manager is not able to work with provided
            CacheStorageContext. When possible we should log message instead, since
            this exception will be propagated to the user.

        Notes
        -----
        Threading: Should be safe to call from any thread.
        