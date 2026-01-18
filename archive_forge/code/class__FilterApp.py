from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class _FilterApp(_ObjectType):
    name = 'filter_app'

    def invoke(self, context):
        next_app = context.next_context.create()
        filter = context.filter_context.create()
        return filter(next_app)