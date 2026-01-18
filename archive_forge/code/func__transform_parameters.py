import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
def _transform_parameters(self, model, params, transformation, target_shape):
    type_name = model.type_name
    if type_name in ['structure', 'map', 'list']:
        getattr(self, '_transform_%s' % type_name)(model, params, transformation, target_shape)