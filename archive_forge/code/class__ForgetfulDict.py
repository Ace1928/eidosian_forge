import copy
from boto3.compat import collections_abc
from boto3.docs.utils import DocumentModifiedShape
from boto3.dynamodb.conditions import ConditionBase, ConditionExpressionBuilder
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
class _ForgetfulDict(dict):
    """A dictionary that discards any items set on it. For use as `memo` in
    `copy.deepcopy()` when every instance of a repeated object in the deepcopied
    data structure should result in a separate copy.
    """

    def __setitem__(self, key, value):
        pass