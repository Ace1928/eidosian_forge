import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
class InvalidPage(Exception):
    pass