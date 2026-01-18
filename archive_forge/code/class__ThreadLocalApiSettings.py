import threading
from typing import Dict, Optional
class _ThreadLocalApiSettings(threading.local):
    api_key: Optional[str]
    cookies: Optional[Dict]
    headers: Optional[Dict]

    def __init__(self) -> None:
        self.api_key = None
        self.cookies = None
        self.headers = None