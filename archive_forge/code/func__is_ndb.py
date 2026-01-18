import cgi
import json
import logging
import os
import pickle
import threading
from google.appengine.api import app_identity
from google.appengine.api import memcache
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext.webapp.util import login_required
import webapp2 as webapp
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import xsrfutil
def _is_ndb(self):
    """Determine whether the model of the instance is an NDB model.

        Returns:
            Boolean indicating whether or not the model is an NDB or DB model.
        """
    if isinstance(self._model, type):
        if _NDB_MODEL is not None and issubclass(self._model, _NDB_MODEL):
            return True
        elif issubclass(self._model, db.Model):
            return False
    raise TypeError('Model class not an NDB or DB model: {0}.'.format(self._model))