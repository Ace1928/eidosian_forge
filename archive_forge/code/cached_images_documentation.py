import queue
import threading
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
from glance import image_cache
import glance.notifier

        PUT /cache/<IMAGE_ID>

        Queues an image for caching. We do not check to see if
        the image is in the registry here. That is done by the
        prefetcher...
        