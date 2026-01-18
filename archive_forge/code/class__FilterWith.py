from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class _FilterWith(_App):
    name = 'filtered_with'

    def invoke(self, context):
        filter = context.filter_context.create()
        filtered = context.next_context.create()
        if context.next_context.object_type is APP:
            return filter(filtered)
        else:

            def composed(app):
                return filter(filtered(app))
            return composed