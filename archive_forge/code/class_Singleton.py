import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
class Singleton(type):
    _instances: typing.Dict[typing.Type['Singleton'], 'Singleton'] = {}

    def __call__(cls, *args, **kwargs):
        instances = cls._instances
        if cls not in instances:
            instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return instances[cls]