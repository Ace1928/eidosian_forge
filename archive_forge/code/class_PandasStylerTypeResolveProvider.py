import sys
from _pydevd_bundle.pydevd_constants import PANDAS_MAX_ROWS, PANDAS_MAX_COLS, PANDAS_MAX_COLWIDTH
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_resolver import inspect, MethodWrapperType
from _pydevd_bundle.pydevd_utils import Timer
from .pydevd_helpers import find_mod_attr
from contextlib import contextmanager
class PandasStylerTypeResolveProvider(object):

    def can_provide(self, type_object, type_name):
        series_class = find_mod_attr('pandas.io.formats.style', 'Styler')
        return series_class is not None and issubclass(type_object, series_class)

    def resolve(self, obj, attribute):
        return getattr(obj, attribute)

    def get_dictionary(self, obj):
        replacements = {'data': '<Styler data -- debugger:skipped eval>', '__dict__': '<dict -- debugger: skipped eval>'}
        return _get_dictionary(obj, replacements)