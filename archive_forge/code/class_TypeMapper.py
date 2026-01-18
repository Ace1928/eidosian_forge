import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
class TypeMapper(dict):

    def __getitem__(self, item):
        options = item.split('|')
        for op in options[:-1]:
            try:
                value = eval_with_catch(op, dict(self.items()))
                break
            except (NameError, KeyError):
                pass
        else:
            value = eval(options[-1], dict(self.items()))
        if value is None:
            return ''
        else:
            return str(value)