import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
def _auto_yaqlize(value, settings):
    if not settings['autoYaqlizeResult']:
        return
    if isinstance(value, type):
        cls = value
    else:
        cls = type(value)
    if cls.__module__ == int.__module__:
        return
    try:
        yaqlization.yaqlize(value, auto_yaqlize_result=True)
    except Exception:
        pass