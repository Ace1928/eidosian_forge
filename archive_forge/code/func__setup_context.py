import os.path
import pkg_resources
from yaql.language import contexts
from yaql.language import conventions
from yaql.language import factory
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql.standard_library import boolean as std_boolean
from yaql.standard_library import branching as std_branching
from yaql.standard_library import collections as std_collections
from yaql.standard_library import common as std_common
from yaql.standard_library import date_time as std_datetime
from yaql.standard_library import math as std_math
from yaql.standard_library import queries as std_queries
from yaql.standard_library import regex as std_regex
from yaql.standard_library import strings as std_strings
from yaql.standard_library import system as std_system
from yaql.standard_library import yaqlized as std_yaqlized
def _setup_context(data, context, finalizer, convention):
    if context is None:
        context = contexts.Context(convention=convention or conventions.CamelCaseConvention())
    if finalizer is None:

        @specs.parameter('iterator', yaqltypes.Iterable())
        @specs.name('#iter')
        def limit(iterator):
            return iterator

        @specs.inject('limiter', yaqltypes.Delegate('#iter'))
        @specs.inject('engine', yaqltypes.Engine())
        @specs.name('#finalize')
        def finalize(obj, limiter, engine):
            if engine.options.get('yaql.convertOutputData', True):
                return utils.convert_output_data(obj, limiter, engine)
            return obj
        context.register_function(limit)
        context.register_function(finalize)
    else:
        context.register_function(finalizer)
    if data is not utils.NO_VALUE:
        context['$'] = utils.convert_input_data(data)
    return context