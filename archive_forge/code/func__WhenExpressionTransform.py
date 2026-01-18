from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import yaml
def _WhenExpressionTransform(when_expression):
    if 'operator' in when_expression:
        when_expression['expressionOperator'] = input_util.CamelToSnake(when_expression.pop('operator')).upper()