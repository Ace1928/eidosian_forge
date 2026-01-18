from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import os
import subprocess
from containerregistry.client import docker_name
def _ExtractProtectedRegion(signature):
    """Extract the length and encoded suffix denoting the protected region."""
    protected = json.loads(_JoseBase64UrlDecode(signature['protected']))
    return (protected['formatLength'], protected['formatTail'])