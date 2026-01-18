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
def _collect_properties(self, content, view_ref, obj_type, path_set=None, include_mors=False):
    """
        Collect properties for managed objects from a view ref
        Check the vSphere API documentation for example on retrieving
        object properties:
            - http://goo.gl/erbFDz
        Args:
            content     (ServiceInstance): ServiceInstance content
            view_ref (pyVmomi.vim.view.*): Starting point of inventory
                                           navigation
            obj_type      (pyVmomi.vim.*): Type of managed object
            path_set               (list): List of properties to retrieve
            include_mors           (bool): If True include the managed objects
                                        refs in the result
        Returns:
            A list of properties for the managed objects
        """
    collector = content.propertyCollector
    obj_spec = vmodl.query.PropertyCollector.ObjectSpec()
    obj_spec.obj = view_ref
    obj_spec.skip = True
    traversal_spec = vmodl.query.PropertyCollector.TraversalSpec()
    traversal_spec.name = 'traverseEntities'
    traversal_spec.path = 'view'
    traversal_spec.skip = False
    traversal_spec.type = view_ref.__class__
    obj_spec.selectSet = [traversal_spec]
    property_spec = vmodl.query.PropertyCollector.PropertySpec()
    property_spec.type = obj_type
    if not path_set:
        property_spec.all = True
    property_spec.pathSet = path_set
    filter_spec = vmodl.query.PropertyCollector.FilterSpec()
    filter_spec.objectSet = [obj_spec]
    filter_spec.propSet = [property_spec]
    props = collector.RetrieveContents([filter_spec])
    data = []
    for obj in props:
        properties = {}
        for prop in obj.propSet:
            properties[prop.name] = prop.val
        if include_mors:
            properties['obj'] = obj.obj
        data.append(properties)
    return data