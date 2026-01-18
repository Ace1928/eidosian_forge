import functools
import inspect
import sys
import pecan
import wsme
import wsme.rest.args
import wsme.rest.json
import wsme.rest.xml
from wsme.utils import is_valid_code
class XMLRenderer(object):

    @staticmethod
    def __init__(path, extra_vars):
        pass

    @staticmethod
    def render(template_path, namespace):
        if 'faultcode' in namespace:
            return wsme.rest.xml.encode_error(None, namespace)
        return wsme.rest.xml.encode_result(namespace['result'], namespace['datatype'])