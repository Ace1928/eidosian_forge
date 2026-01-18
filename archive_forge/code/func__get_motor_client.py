from typing import Any, Dict, List, Optional, Sequence
from langchain_community.indexes.base import RecordManager
def _get_motor_client(mongodb_url: str, **kwargs: Any) -> Any:
    """Get AsyncIOMotorClient for async operations from the mongodb_url,
    otherwise raise error."""
    try:
        motor = _import_motor_asyncio()
        client = motor(mongodb_url, **kwargs)
    except ValueError as e:
        raise ImportError(f'AsyncIOMotorClient string provided is not in proper format. Got error: {e} ')
    return client