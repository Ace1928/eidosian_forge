import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
url with user:password part removed unless it is formed with
        environment variables as specified in PEP 610, or it is ``git``
        in the case of a git URL.
        