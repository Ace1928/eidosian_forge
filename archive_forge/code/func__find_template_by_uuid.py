import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _find_template_by_uuid(self, template_uuid):
    try:
        template = self.find_by_uuid(template_uuid)
    except LibcloudError:
        content = self.connection.RetrieveContent()
        vms = content.viewManager.CreateContainerView(content.rootFolder, [vim.VirtualMachine], recursive=True).view
        for vm in vms:
            if vm.config.instanceUuid == template_uuid:
                template = vm
    except Exception as exc:
        raise LibcloudError('Error while searching for template: %s' % exc, driver=self)
    if not template:
        raise LibcloudError('Unable to locate VirtualMachine.', driver=self)
    return template