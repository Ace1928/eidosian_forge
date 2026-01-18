from __future__ import annotations
from typing import Any, Tuple
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores.redis.filters import (
from langchain_community.vectorstores.redis.schema import RedisModel
from langchain.chains.query_constructor.ir import (
def _attribute_to_filter_field(self, attribute: str) -> RedisFilterField:
    if attribute in [tf.name for tf in self._schema.text]:
        return RedisText(attribute)
    elif attribute in [tf.name for tf in self._schema.tag or []]:
        return RedisTag(attribute)
    elif attribute in [tf.name for tf in self._schema.numeric or []]:
        return RedisNum(attribute)
    else:
        raise ValueError(f'Invalid attribute {attribute} not in vector store schema. Schema is:\n{self._schema.as_dict()}')