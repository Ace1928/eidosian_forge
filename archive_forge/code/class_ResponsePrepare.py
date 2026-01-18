import asyncio
import functools
import queue
import threading
import time
from typing import (
class ResponsePrepare(NamedTuple):
    birth_artifact_id: str
    upload_url: Optional[str]
    upload_headers: Sequence[str]
    upload_id: Optional[str]
    storage_path: Optional[str]
    multipart_upload_urls: Optional[Dict[int, str]]