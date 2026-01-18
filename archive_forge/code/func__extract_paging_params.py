import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _extract_paging_params(self, kwargs):
    pagination_config = kwargs.pop('PaginationConfig', {})
    max_items = pagination_config.get('MaxItems', None)
    if max_items is not None:
        max_items = int(max_items)
    page_size = pagination_config.get('PageSize', None)
    if page_size is not None:
        if self._limit_key is None:
            raise PaginationError(message='PageSize parameter is not supported for the pagination interface for this operation.')
        input_members = self._model.input_shape.members
        limit_key_shape = input_members.get(self._limit_key)
        if limit_key_shape.type_name == 'string':
            if not isinstance(page_size, str):
                page_size = str(page_size)
        else:
            page_size = int(page_size)
    return {'MaxItems': max_items, 'StartingToken': pagination_config.get('StartingToken', None), 'PageSize': page_size}