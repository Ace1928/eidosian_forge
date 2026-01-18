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
class CredentialsProperty(db.Property):
    """App Engine datastore Property for Credentials.

    Utility property that allows easy storage and retrieval of
    oauth2client.Credentials
    """
    data_type = client.Credentials

    def get_value_for_datastore(self, model_instance):
        logger.info('get: Got type ' + str(type(model_instance)))
        cred = super(CredentialsProperty, self).get_value_for_datastore(model_instance)
        if cred is None:
            cred = ''
        else:
            cred = cred.to_json()
        return db.Blob(cred)

    def make_value_from_datastore(self, value):
        logger.info('make: Got type ' + str(type(value)))
        if value is None:
            return None
        if len(value) == 0:
            return None
        try:
            credentials = client.Credentials.new_from_json(value)
        except ValueError:
            credentials = None
        return credentials

    def validate(self, value):
        value = super(CredentialsProperty, self).validate(value)
        logger.info('validate: Got type ' + str(type(value)))
        if value is not None and (not isinstance(value, client.Credentials)):
            raise db.BadValueError('Property {0} must be convertible to a Credentials instance ({1})'.format(self.name, value))
        return value