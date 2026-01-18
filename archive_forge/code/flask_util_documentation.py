from functools import wraps
import hashlib
import json
import os
import pickle
import six.moves.http_client as httplib
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
Returns an authorized http instance.

        Can only be called if there are valid credentials for the user, such
        as inside of a view that is decorated with @required.

        Args:
            *args: Positional arguments passed to httplib2.Http constructor.
            **kwargs: Positional arguments passed to httplib2.Http constructor.

        Raises:
            ValueError if no credentials are available.
        