import os
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import failure
from glance.common import exception
from glance.i18n import _, _LE
Create temp file into store and return path to it

        :param image_id: Glance Image ID
        