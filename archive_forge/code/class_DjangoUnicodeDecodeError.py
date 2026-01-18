import codecs
import datetime
import locale
from decimal import Decimal
from types import NoneType
from urllib.parse import quote
from django.utils.functional import Promise
class DjangoUnicodeDecodeError(UnicodeDecodeError):

    def __init__(self, obj, *args):
        self.obj = obj
        super().__init__(*args)

    def __str__(self):
        return '%s. You passed in %r (%s)' % (super().__str__(), self.obj, type(self.obj))