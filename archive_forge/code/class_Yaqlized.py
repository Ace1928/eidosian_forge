import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
class Yaqlized(yaqltypes.GenericType):

    def __init__(self, can_access_attributes=False, can_call_methods=False, can_index=False):

        def check_value(value, context, *args, **kwargs):
            settings = yaqlization.get_yaqlization_settings(value)
            if settings is None:
                return False
            if can_access_attributes and (not settings['yaqlizeAttributes']):
                return False
            if can_call_methods and (not settings['yaqlizeMethods']):
                return False
            if can_index and (not settings['yaqlizeIndexer']):
                return False
            return True
        super(Yaqlized, self).__init__(checker=check_value, nullable=False)