import argparse
import logging
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import save
from containerregistry.tools import logging_setup
from containerregistry.tools import patched
This package imports images from a 'docker save' tarball.

Unlike 'docker save' the format this uses is proprietary.
