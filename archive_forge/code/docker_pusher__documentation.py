from __future__ import absolute_import
from __future__ import print_function
import argparse
import logging
import sys
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_session
from containerregistry.client.v2_2 import oci_compat
from containerregistry.tools import logging_setup
from containerregistry.tools import patched
from containerregistry.transport import retry
from containerregistry.transport import transport_pool
import httplib2
Perform substitutions in the provided tag name.