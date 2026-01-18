import json
import os
import re
import uuid
from urllib.parse import urlencode
import tornado.auth
import tornado.gen
import tornado.web
from celery.utils.imports import instantiate
from tornado.options import options
from ..views import BaseHandler
from ..views.error import NotFoundErrorHandler
class LoginHandler(BaseHandler):

    def __new__(cls, *args, **kwargs):
        return instantiate(options.auth_provider or NotFoundErrorHandler, *args, **kwargs)