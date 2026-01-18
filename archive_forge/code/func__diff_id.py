from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import errno
import io
import json
import os
import tarfile
import concurrent.futures
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_image as v1_image
from containerregistry.client.v1 import save as v1_save
from containerregistry.client.v2 import v1_compat
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import v2_compat
import six
def _diff_id(v1_img, blob):
    try:
        return v1_img.diff_id(blob)
    except ValueError:
        unzipped = v1_img.uncompressed_layer(blob)
        return docker_digest.SHA256(unzipped)