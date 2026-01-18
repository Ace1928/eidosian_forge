from collections import Counter
from ...error import GraphQLError
from ...pyutils.ordereddict import OrderedDict
from ...type.definition import (GraphQLInterfaceType, GraphQLObjectType,
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
class OrderedCounter(Counter, OrderedDict):
    pass