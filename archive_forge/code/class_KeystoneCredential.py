import collections
import configparser
import re
from oslo_log import log as logging
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import eventfactory as factory
from pycadf import host
from pycadf import identifier
from pycadf import resource
from pycadf import tag
from urllib import parse as urlparse
class KeystoneCredential(credential.Credential):

    def __init__(self, identity_status=None, **kwargs):
        super(KeystoneCredential, self).__init__(**kwargs)
        if identity_status is not None:
            self.identity_status = identity_status