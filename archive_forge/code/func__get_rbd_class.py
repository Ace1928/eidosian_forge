from typing import Optional
from oslo_utils import fileutils
from oslo_utils import importutils
import os_brick.privileged
def _get_rbd_class():
    global RBDConnector
    global get_rbd_class
    if not RBDConnector:
        rbd_class_route = 'os_brick.initiator.connectors.rbd.RBDConnector'
        RBDConnector = importutils.import_class(rbd_class_route)
    get_rbd_class = lambda: None