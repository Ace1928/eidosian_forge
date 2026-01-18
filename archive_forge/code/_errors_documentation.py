import re
from typing import Optional
from requests import HTTPError, Response
from ._fixes import JSONDecodeError
Append additional information to the `HfHubHTTPError` initial message.