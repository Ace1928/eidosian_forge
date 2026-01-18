import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
def _transform_map(self, model, params, transformation, target_shape):
    if not isinstance(params, collections_abc.Mapping):
        return
    value_model = model.value
    value_shape = value_model.name
    for key, value in params.items():
        if value_shape == target_shape:
            params[key] = transformation(value)
        else:
            self._transform_parameters(value_model, params[key], transformation, target_shape)