import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
class Mgmt(object):

    def __init__(self, dbaas):
        self.instances = dbaas.management
        self.hosts = dbaas.hosts
        self.accounts = dbaas.accounts
        self.storage = dbaas.storage
        self.datastore_version = dbaas.mgmt_datastore_versions