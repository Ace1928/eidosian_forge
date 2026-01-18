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
@property
def _OAUTH_ACCESS_TOKEN_URL(self):
    return f'{self.base_url}/v1/token'