from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import os
import subprocess
from containerregistry.client import docker_name
class BadManifestException(Exception):
    """Exception type raised when a malformed manifest is encountered."""