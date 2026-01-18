from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import TextField
from django.db.models.fields.json import JSONField
from django.utils.regex_helper import _lazy_re_compile
class ArgJoiner:

    def join(self, args):
        args = [' VALUE '.join(arg) for arg in zip(args[::2], args[1::2])]
        return ', '.join(args)