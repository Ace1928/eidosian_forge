import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
@staticmethod
def _create_exec_configuration(input, **config):
    """
        Prepares the input parameters for executyion API call
        """
    if 'environment' in config.keys():
        input['environment'] = config['environment']
    if 'width' in config.keys():
        input['width'] = int(config['width'])
    else:
        input['width'] = 80
    if 'height' in config.keys():
        input['height'] = int(config['height'])
    else:
        input['height'] = 25
    if 'user' in config.keys():
        input['user'] = config['user']
    if 'group' in config.keys():
        input['group'] = config['group']
    if 'cwd' in config.keys():
        input['cwd'] = config['cwd']
    if 'wait-for-websocket' in config.keys():
        input['wait-for-websocket'] = config['wait-for-websocket']
    else:
        input['wait-for-websocket'] = False
    if 'record-output' in config.keys():
        input['record-output'] = config['record-output']
    if 'interactive' in config.keys():
        input['interactive'] = config['interactive']
    return input