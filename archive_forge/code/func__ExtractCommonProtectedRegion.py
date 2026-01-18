from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import os
import subprocess
from containerregistry.client import docker_name
def _ExtractCommonProtectedRegion(signatures):
    """Verify that the signatures agree on the protected region and return one."""
    p = _ExtractProtectedRegion(signatures[0])
    for sig in signatures[1:]:
        if p != _ExtractProtectedRegion(sig):
            raise BadManifestException('Signatures disagree on protected region')
    return p