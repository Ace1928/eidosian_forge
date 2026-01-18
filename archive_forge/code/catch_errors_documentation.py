import logging
import re
import webob.dec
import webob.exc
from oslo_middleware import base
Middleware that provides high-level error handling.

    It catches all exceptions from subsequent applications in WSGI pipeline
    to hide internal errors from API response.
    