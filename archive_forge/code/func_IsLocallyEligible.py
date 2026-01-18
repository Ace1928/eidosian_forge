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
def IsLocallyEligible(self):
    return True