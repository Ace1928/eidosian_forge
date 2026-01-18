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
def _delete_entity(self):
    """Delete entity from datastore.

        Attempts to delete using the key_name stored on the object, whether or
        not the given key is in the datastore.
        """
    if self._is_ndb():
        _NDB_KEY(self._model, self._key_name).delete()
    else:
        entity_key = db.Key.from_path(self._model.kind(), self._key_name)
        db.delete(entity_key)