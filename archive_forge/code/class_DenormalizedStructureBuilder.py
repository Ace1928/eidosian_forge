from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
class DenormalizedStructureBuilder:
    """Build a StructureShape from a denormalized model.

    This is a convenience builder class that makes it easy to construct
    ``StructureShape``s based on a denormalized model.

    It will handle the details of creating unique shape names and creating
    the appropriate shape map needed by the ``StructureShape`` class.

    Example usage::

        builder = DenormalizedStructureBuilder()
        shape = builder.with_members({
            'A': {
                'type': 'structure',
                'members': {
                    'B': {
                        'type': 'structure',
                        'members': {
                            'C': {
                                'type': 'string',
                            }
                        }
                    }
                }
            }
        }).build_model()
        # ``shape`` is now an instance of botocore.model.StructureShape

    :type dict_type: class
    :param dict_type: The dictionary type to use, allowing you to opt-in
                      to using OrderedDict or another dict type. This can
                      be particularly useful for testing when order
                      matters, such as for documentation.

    """
    SCALAR_TYPES = ('string', 'integer', 'boolean', 'blob', 'float', 'timestamp', 'long', 'double', 'char')

    def __init__(self, name=None):
        self.members = OrderedDict()
        self._name_generator = ShapeNameGenerator()
        if name is None:
            self.name = self._name_generator.new_shape_name('structure')

    def with_members(self, members):
        """

        :type members: dict
        :param members: The denormalized members.

        :return: self

        """
        self._members = members
        return self

    def build_model(self):
        """Build the model based on the provided members.

        :rtype: botocore.model.StructureShape
        :return: The built StructureShape object.

        """
        shapes = OrderedDict()
        denormalized = {'type': 'structure', 'members': self._members}
        self._build_model(denormalized, shapes, self.name)
        resolver = ShapeResolver(shape_map=shapes)
        return StructureShape(shape_name=self.name, shape_model=shapes[self.name], shape_resolver=resolver)

    def _build_model(self, model, shapes, shape_name):
        if model['type'] == 'structure':
            shapes[shape_name] = self._build_structure(model, shapes)
        elif model['type'] == 'list':
            shapes[shape_name] = self._build_list(model, shapes)
        elif model['type'] == 'map':
            shapes[shape_name] = self._build_map(model, shapes)
        elif model['type'] in self.SCALAR_TYPES:
            shapes[shape_name] = self._build_scalar(model)
        else:
            raise InvalidShapeError(f'Unknown shape type: {model['type']}')

    def _build_structure(self, model, shapes):
        members = OrderedDict()
        shape = self._build_initial_shape(model)
        shape['members'] = members
        for name, member_model in model.get('members', OrderedDict()).items():
            member_shape_name = self._get_shape_name(member_model)
            members[name] = {'shape': member_shape_name}
            self._build_model(member_model, shapes, member_shape_name)
        return shape

    def _build_list(self, model, shapes):
        member_shape_name = self._get_shape_name(model)
        shape = self._build_initial_shape(model)
        shape['member'] = {'shape': member_shape_name}
        self._build_model(model['member'], shapes, member_shape_name)
        return shape

    def _build_map(self, model, shapes):
        key_shape_name = self._get_shape_name(model['key'])
        value_shape_name = self._get_shape_name(model['value'])
        shape = self._build_initial_shape(model)
        shape['key'] = {'shape': key_shape_name}
        shape['value'] = {'shape': value_shape_name}
        self._build_model(model['key'], shapes, key_shape_name)
        self._build_model(model['value'], shapes, value_shape_name)
        return shape

    def _build_initial_shape(self, model):
        shape = {'type': model['type']}
        if 'documentation' in model:
            shape['documentation'] = model['documentation']
        for attr in Shape.METADATA_ATTRS:
            if attr in model:
                shape[attr] = model[attr]
        return shape

    def _build_scalar(self, model):
        return self._build_initial_shape(model)

    def _get_shape_name(self, model):
        if 'shape_name' in model:
            return model['shape_name']
        else:
            return self._name_generator.new_shape_name(model['type'])