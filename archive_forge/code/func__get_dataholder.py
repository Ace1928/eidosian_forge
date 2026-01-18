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
def _get_dataholder(self, instance):
    dataholder = getattr(instance, '_wsme_dataholder', None)
    if dataholder is None:
        dataholder = instance._wsme_DataHolderClass()
        instance._wsme_dataholder = dataholder
    return dataholder