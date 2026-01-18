import copy
from boto3.compat import collections_abc
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from boto3.dynamodb.conditions import ConditionBase
from boto3.dynamodb.conditions import ConditionExpressionBuilder
from boto3.docs.utils import DocumentModifiedShape
class DynamoDBHighLevelResource(object):

    def __init__(self, *args, **kwargs):
        super(DynamoDBHighLevelResource, self).__init__(*args, **kwargs)
        self.meta.client.meta.events.register('provide-client-params.dynamodb', copy_dynamodb_params, unique_id='dynamodb-create-params-copy')
        self._injector = TransformationInjector()
        self.meta.client.meta.events.register('before-parameter-build.dynamodb', self._injector.inject_condition_expressions, unique_id='dynamodb-condition-expression')
        self.meta.client.meta.events.register('before-parameter-build.dynamodb', self._injector.inject_attribute_value_input, unique_id='dynamodb-attr-value-input')
        self.meta.client.meta.events.register('after-call.dynamodb', self._injector.inject_attribute_value_output, unique_id='dynamodb-attr-value-output')
        attr_value_shape_docs = DocumentModifiedShape('AttributeValue', new_type='valid DynamoDB type', new_description='- The value of the attribute. The valid value types are listed in the :ref:`DynamoDB Reference Guide<ref_valid_dynamodb_types>`.', new_example_value="'string'|123|Binary(b'bytes')|True|None|set(['string'])|set([123])|set([Binary(b'bytes')])|[]|{}")
        key_expression_shape_docs = DocumentModifiedShape('KeyExpression', new_type='condition from :py:class:`boto3.dynamodb.conditions.Key` method', new_description='The condition(s) a key(s) must meet. Valid conditions are listed in the :ref:`DynamoDB Reference Guide<ref_dynamodb_conditions>`.', new_example_value="Key('mykey').eq('myvalue')")
        con_expression_shape_docs = DocumentModifiedShape('ConditionExpression', new_type='condition from :py:class:`boto3.dynamodb.conditions.Attr` method', new_description='The condition(s) an attribute(s) must meet. Valid conditions are listed in the :ref:`DynamoDB Reference Guide<ref_dynamodb_conditions>`.', new_example_value="Attr('myattribute').eq('myvalue')")
        self.meta.client.meta.events.register('docs.*.dynamodb.*.complete-section', attr_value_shape_docs.replace_documentation_for_matching_shape, unique_id='dynamodb-attr-value-docs')
        self.meta.client.meta.events.register('docs.*.dynamodb.*.complete-section', key_expression_shape_docs.replace_documentation_for_matching_shape, unique_id='dynamodb-key-expression-docs')
        self.meta.client.meta.events.register('docs.*.dynamodb.*.complete-section', con_expression_shape_docs.replace_documentation_for_matching_shape, unique_id='dynamodb-cond-expression-docs')