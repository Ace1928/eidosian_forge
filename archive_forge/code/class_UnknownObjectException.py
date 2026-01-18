import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
class UnknownObjectException(GithubException):
    """
    Exception raised when a non-existing object is requested (when Github API replies with a 404 HTML status)
    """