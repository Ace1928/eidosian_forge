import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
def _remap_name(name, settings):
    return settings['attributeRemapping'].get(name, name)