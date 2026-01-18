import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
class IncompletableObject(GithubException):
    """
    Exception raised when we can not request an object from Github because the data returned did not include a URL
    """