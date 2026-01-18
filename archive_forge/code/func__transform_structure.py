import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
def _transform_structure(self, model, params, transformation, target_shape):
    if not isinstance(params, collections_abc.Mapping):
        return
    for param in params:
        if param in model.members:
            member_model = model.members[param]
            member_shape = member_model.name
            if member_shape == target_shape:
                params[param] = transformation(params[param])
            else:
                self._transform_parameters(member_model, params[param], transformation, target_shape)