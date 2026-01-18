from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
class ShapeResolver:
    """Resolves shape references."""
    SHAPE_CLASSES = {'structure': StructureShape, 'list': ListShape, 'map': MapShape, 'string': StringShape}

    def __init__(self, shape_map):
        self._shape_map = shape_map
        self._shape_cache = {}

    def get_shape_by_name(self, shape_name, member_traits=None):
        try:
            shape_model = self._shape_map[shape_name]
        except KeyError:
            raise NoShapeFoundError(shape_name)
        try:
            shape_cls = self.SHAPE_CLASSES.get(shape_model['type'], Shape)
        except KeyError:
            raise InvalidShapeError(f"Shape is missing required key 'type': {shape_model}")
        if member_traits:
            shape_model = shape_model.copy()
            shape_model.update(member_traits)
        result = shape_cls(shape_name, shape_model, self)
        return result

    def resolve_shape_ref(self, shape_ref):
        if len(shape_ref) == 1 and 'shape' in shape_ref:
            return self.get_shape_by_name(shape_ref['shape'])
        else:
            member_traits = shape_ref.copy()
            try:
                shape_name = member_traits.pop('shape')
            except KeyError:
                raise InvalidShapeReferenceError(f'Invalid model, missing shape reference: {shape_ref}')
            return self.get_shape_by_name(shape_name, member_traits)