import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class CalculatedProperty(Property):

    def __init__(self, verbose_name=None, name=None, default=None, required=False, validator=None, choices=None, calculated_type=int, unique=False, use_method=False):
        super(CalculatedProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)
        self.calculated_type = calculated_type
        self.use_method = use_method

    def __get__(self, obj, objtype):
        value = self.default_value()
        if obj:
            try:
                value = getattr(obj, self.slot_name)
                if self.use_method:
                    value = value()
            except AttributeError:
                pass
        return value

    def __set__(self, obj, value):
        """Not possible to set a new AutoID."""
        pass

    def _set_direct(self, obj, value):
        if not self.use_method:
            setattr(obj, self.slot_name, value)

    def get_value_for_datastore(self, model_instance):
        if self.calculated_type in [str, int, bool]:
            value = self.__get__(model_instance, model_instance.__class__)
            return value
        else:
            return None