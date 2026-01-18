from __future__ import absolute_import
import io
import logging
import os
import random
import re
import time
import urllib
import httplib2
from oauth2client import client
from oauth2client import file as oauth2client_file
from oauth2client import tools
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.tools.value_mixin import ValueMixin
from googlecloudsdk.third_party.appengine._internal import six_subset
class FlowFlags(object):

    def __init__(self, options):
        self.logging_level = logging.getLevelName(logging.getLogger().level)
        self.noauth_local_webserver = not options.auth_local_webserver if options else True
        self.auth_host_port = [8080, 8090]
        self.auth_host_name = 'localhost'