import logging
import socket
import socketserver
import sys
from collections import deque
from wsgiref import simple_server
from django.core.exceptions import ImproperlyConfigured
from django.core.handlers.wsgi import LimitedStream
from django.core.wsgi import get_wsgi_application
from django.db import connections
from django.utils.module_loading import import_string
Copy of WSGIRequestHandler.handle() but with different ServerHandler