from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import base64
import errno
import io
import json
import logging
import os
import subprocess
from containerregistry.client import docker_name
import httplib2
from oauth2client import client as oauth2client
import six
class Anonymous(Provider):
    """Implementation for anonymous access."""

    def Get(self):
        """Implement anonymous authentication."""
        return ''