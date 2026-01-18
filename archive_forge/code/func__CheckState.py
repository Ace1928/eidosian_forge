from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import re
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_creds as v2_2_creds
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
def _CheckState(predicate, message=None):
    if not predicate:
        raise BadStateException(message if message else 'Unknown')