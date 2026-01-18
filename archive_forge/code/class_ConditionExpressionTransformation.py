import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
class ConditionExpressionTransformation(object):
    """Provides a transformation for condition expressions

    The ``ParameterTransformer`` class can call this class directly
    to transform the condition expressions in the parameters provided.
    """

    def __init__(self, condition_builder, placeholder_names, placeholder_values, is_key_condition=False):
        self._condition_builder = condition_builder
        self._placeholder_names = placeholder_names
        self._placeholder_values = placeholder_values
        self._is_key_condition = is_key_condition

    def __call__(self, value):
        if isinstance(value, ConditionBase):
            built_expression = self._condition_builder.build_expression(value, is_key_condition=self._is_key_condition)
            self._placeholder_names.update(built_expression.attribute_name_placeholders)
            self._placeholder_values.update(built_expression.attribute_value_placeholders)
            return built_expression.condition_expression
        return value