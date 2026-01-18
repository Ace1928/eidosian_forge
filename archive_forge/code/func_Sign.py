from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import os
import subprocess
from containerregistry.client import docker_name
def Sign(unsigned_manifest):
    return unsigned_manifest