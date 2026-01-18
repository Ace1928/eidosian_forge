from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
class CallbackRequestSchema(BaseRequest):
    """
    Parameters for Callbacks upon Completion
    """
    callback_url: Optional[str] = Field(None, description='If provided, will send a callback notification to this endpoint when the job is complete', exclude=True)
    callback_id: Optional[str] = Field(None, description='If provided, will use this callback id when sending a callback notification to the callback_url', exclude=True)
    callback_method: Optional[str] = Field('POST', description='If provided, will use this method when sending a callback notification to the callback_url', exclude=True)
    callback_params: Optional[Dict[str, Any]] = Field(None, description='If provided, will use these parameters when sending a callback notification to the callback_url', exclude=True)
    callback_headers: Optional[Dict[str, Any]] = Field(None, description='If provided, will use these headers when sending a callback notification to the callback_url', exclude=True)
    callback_retries: Optional[int] = Field(None, description='If provided, will retry the callback this many times before giving up', exclude=True)
    callback_timeout: Optional[int] = Field(None, description='If provided, will timeout the callback after this many seconds', exclude=True)
    callback_refire: Optional[bool] = Field(None, description='If provided, will refire the callback when the result is retrieved again', exclude=True)

    @property
    def callback_enabled(self) -> bool:
        """
        Returns True if callback_url is not None
        """
        return self.callback_url is not None

    @property
    def callback_param_fields(self) -> List[str]:
        """
        Returns the param fields
        """
        return ['callback_url', 'callback_id', 'callback_method', 'callback_params', 'callback_headers', 'callback_retries', 'callback_timeout', 'callback_refire']

    @classmethod
    def from_request(cls, **kwargs) -> 'CallbackRequestSchema':
        """
        Parses the keywords from the request
        """
        from lazyops.utils.helpers import build_dict_from_query
        if (callback_params := kwargs.pop('callback_params', None)):
            kwargs['callback_params'] = build_dict_from_query(callback_params)
        if (callback_headers := kwargs.pop('callback_headers', None)):
            kwargs['callback_headers'] = build_dict_from_query(callback_headers)
        return cls(**kwargs)

    def to_request_kwargs(self, exclude_none: bool=True, exclude_unset: bool=True, **kwargs) -> Dict[str, Any]:
        """
        Returns the request kwargs
        """
        return self.model_dump(include=set(self.callback_param_fields), exclude_none=exclude_none, exclude_unset=exclude_unset, **kwargs)