import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _json_to_treespec(json_schema: DumpableContext) -> TreeSpec:
    if json_schema['type'] is None and json_schema['context'] is None and (len(json_schema['children_spec']) == 0):
        return LeafSpec()
    if json_schema['type'] not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(f'Deserializing {json_schema['type']} in pytree is not registered.')
    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[json_schema['type']]
    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[typ]
    if serialize_node_def.from_dumpable_context is None:
        try:
            context = json.loads(json_schema['context'])
        except TypeError as ex:
            raise TypeError('Unable to deserialize context. Please make the context json load-able, or register a custom serializer using _register_pytree_node.') from ex
    else:
        context = serialize_node_def.from_dumpable_context(json_schema['context'])
    children_spec = []
    for child_string in json_schema['children_spec']:
        children_spec.append(_json_to_treespec(child_string))
    return TreeSpec(typ, context, children_spec)