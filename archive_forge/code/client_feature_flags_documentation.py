import json
import urllib.parse
from tensorboard import context
from tensorboard import errors
Initializes this middleware.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        