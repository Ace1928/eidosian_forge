import base64
import binascii
import logging
import bcrypt
import webob
from oslo_config import cfg
from oslo_middleware import base
Parse WSGI environment for Authorization header of type Basic

    :param: env: WSGI environment to get header from
    :returns: Token portion of the header value
    :raises: HTTPUnauthorized, if header is missing or if the type is not Basic
    