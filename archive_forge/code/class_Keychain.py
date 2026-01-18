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
class Keychain(six.with_metaclass(abc.ABCMeta, object)):
    """Interface for resolving an image reference to a credential."""

    @abc.abstractmethod
    def Resolve(self, name):
        """Resolves the appropriate credential for the given registry.

    Args:
      name: the registry for which we need a credential.

    Returns:
      a Provider suitable for use with registry operations.
    """