from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
def _ScopesFromMetadataServer(self, scopes):
    """Returns instance scopes based on GCE metadata server."""
    if not util.DetectGce():
        raise exceptions.ResourceUnavailableError('GCE credentials requested outside a GCE instance')
    if not self.GetServiceAccount(self.__service_account_name):
        raise exceptions.ResourceUnavailableError('GCE credentials requested but service account %s does not exist.' % self.__service_account_name)
    if scopes:
        scope_ls = util.NormalizeScopes(scopes)
        instance_scopes = self.GetInstanceScopes()
        if scope_ls > instance_scopes:
            raise exceptions.CredentialsError('Instance did not have access to scopes %s' % (sorted(list(scope_ls - instance_scopes)),))
    else:
        scopes = self.GetInstanceScopes()
    return scopes