import base64
import getpass
import json
import logging
import sys
from oauth2client.contrib import reauth_errors
from pyu2f import errors as u2ferrors
from pyu2f import model
from pyu2f.convenience import authenticator
from six.moves import urllib
def InteractiveCheck():
    """Check if we are in an interractive environment.

    If the rapt token needs refreshing, the user needs to answer the challenges.
    If the user is not in an interractive environment, the challenges can not be
    answered and we just wait for timeout for no reason."""
    return sys.stdin.isatty()