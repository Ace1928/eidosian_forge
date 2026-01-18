from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.apigee import base
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import request
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import log
class ProductsClient(base.FieldPagedListClient):
    """REST client for Apigee API products."""
    _entity_path = ['organization', 'product']
    _list_container = 'apiProduct'
    _page_field = 'name'

    @classmethod
    def Create(cls, identifiers, product_info):
        product_dict = product_info._asdict()
        product_dict = {key: product_dict[key] for key in product_dict if product_dict[key] is not None}
        return request.ResponseToApiRequest(identifiers, ['organization'], 'product', method='POST', body=json.dumps(product_dict))

    @classmethod
    def Update(cls, identifiers, product_info):
        product_dict = product_info._asdict()
        product_dict = {key: product_dict[key] for key in product_dict if product_dict[key] is not None}
        return request.ResponseToApiRequest(identifiers, ['organization', 'product'], method='PUT', body=json.dumps(product_dict))