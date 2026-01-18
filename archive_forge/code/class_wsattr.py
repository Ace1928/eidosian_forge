import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
class wsattr(object):
    """
    Complex type attribute definition.

    Example::

        class MyComplexType(wsme.types.Base):
            optionalvalue = int
            mandatoryvalue = wsattr(int, mandatory=True)
            named_value = wsattr(int, name='named.value')

    After inspection, the non-wsattr attributes will be replaced, and
    the above class will be equivalent to::

        class MyComplexType(wsme.types.Base):
            optionalvalue = wsattr(int)
            mandatoryvalue = wsattr(int, mandatory=True)

    """

    def __init__(self, datatype, mandatory=False, name=None, default=Unset, readonly=False):
        self.key = None
        self.name = name
        self._datatype = (datatype,)
        self.mandatory = mandatory
        self.default = default
        self.readonly = readonly
        self.complextype = None

    def _get_dataholder(self, instance):
        dataholder = getattr(instance, '_wsme_dataholder', None)
        if dataholder is None:
            dataholder = instance._wsme_DataHolderClass()
            instance._wsme_dataholder = dataholder
        return dataholder

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(self._get_dataholder(instance), self.key, self.default)

    def __set__(self, instance, value):
        try:
            value = validate_value(self.datatype, value)
        except (ValueError, TypeError) as e:
            raise exc.InvalidInput(self.name, value, str(e))
        dataholder = self._get_dataholder(instance)
        if value is Unset:
            if hasattr(dataholder, self.key):
                delattr(dataholder, self.key)
        else:
            setattr(dataholder, self.key, value)

    def __delete__(self, instance):
        self.__set__(instance, Unset)

    def _get_datatype(self):
        if isinstance(self._datatype, tuple):
            self._datatype = self.complextype().__registry__.resolve_type(self._datatype[0])
        if isinstance(self._datatype, weakref.ref):
            return self._datatype()
        if isinstance(self._datatype, list):
            return [item() if isinstance(item, weakref.ref) else item for item in self._datatype]
        return self._datatype

    def _set_datatype(self, datatype):
        self._datatype = datatype
    datatype = property(_get_datatype, _set_datatype)