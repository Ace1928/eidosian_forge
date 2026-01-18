from __future__ import absolute_import
from __future__ import print_function
import argparse
import logging
import tarfile
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.tools import logging_setup
from six.moves import zip  # pylint: disable=redefined-builtin
This package flattens image metadata into a single tarball.