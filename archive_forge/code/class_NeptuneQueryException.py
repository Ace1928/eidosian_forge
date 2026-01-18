import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
class NeptuneQueryException(Exception):
    """Exception for the Neptune queries."""

    def __init__(self, exception: Union[str, Dict]):
        if isinstance(exception, dict):
            self.message = exception['message'] if 'message' in exception else 'unknown'
            self.details = exception['details'] if 'details' in exception else 'unknown'
        else:
            self.message = exception
            self.details = 'unknown'

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details