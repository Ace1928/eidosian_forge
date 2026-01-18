from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
class _WalkCoreSchema:

    def __init__(self):
        self._schema_type_to_method = self._build_schema_type_to_method()

    def _build_schema_type_to_method(self) -> dict[core_schema.CoreSchemaType, Recurse]:
        mapping: dict[core_schema.CoreSchemaType, Recurse] = {}
        key: core_schema.CoreSchemaType
        for key in get_args(core_schema.CoreSchemaType):
            method_name = f'handle_{key.replace('-', '_')}_schema'
            mapping[key] = getattr(self, method_name, self._handle_other_schemas)
        return mapping

    def walk(self, schema: core_schema.CoreSchema, f: Walk) -> core_schema.CoreSchema:
        return f(schema, self._walk)

    def _walk(self, schema: core_schema.CoreSchema, f: Walk) -> core_schema.CoreSchema:
        schema = self._schema_type_to_method[schema['type']](schema.copy(), f)
        ser_schema: core_schema.SerSchema | None = schema.get('serialization')
        if ser_schema:
            schema['serialization'] = self._handle_ser_schemas(ser_schema, f)
        return schema

    def _handle_other_schemas(self, schema: core_schema.CoreSchema, f: Walk) -> core_schema.CoreSchema:
        sub_schema = schema.get('schema', None)
        if sub_schema is not None:
            schema['schema'] = self.walk(sub_schema, f)
        return schema

    def _handle_ser_schemas(self, ser_schema: core_schema.SerSchema, f: Walk) -> core_schema.SerSchema:
        schema: core_schema.CoreSchema | None = ser_schema.get('schema', None)
        if schema is not None:
            ser_schema['schema'] = self.walk(schema, f)
        return_schema: core_schema.CoreSchema | None = ser_schema.get('return_schema', None)
        if return_schema is not None:
            ser_schema['return_schema'] = self.walk(return_schema, f)
        return ser_schema

    def handle_definitions_schema(self, schema: core_schema.DefinitionsSchema, f: Walk) -> core_schema.CoreSchema:
        new_definitions: list[core_schema.CoreSchema] = []
        for definition in schema['definitions']:
            if 'schema_ref' in definition and 'ref' in definition:
                new_definitions.append(definition)
                self.walk(definition, f)
                continue
            updated_definition = self.walk(definition, f)
            if 'ref' in updated_definition:
                new_definitions.append(updated_definition)
        new_inner_schema = self.walk(schema['schema'], f)
        if not new_definitions and len(schema) == 3:
            return new_inner_schema
        new_schema = schema.copy()
        new_schema['schema'] = new_inner_schema
        new_schema['definitions'] = new_definitions
        return new_schema

    def handle_list_schema(self, schema: core_schema.ListSchema, f: Walk) -> core_schema.CoreSchema:
        items_schema = schema.get('items_schema')
        if items_schema is not None:
            schema['items_schema'] = self.walk(items_schema, f)
        return schema

    def handle_set_schema(self, schema: core_schema.SetSchema, f: Walk) -> core_schema.CoreSchema:
        items_schema = schema.get('items_schema')
        if items_schema is not None:
            schema['items_schema'] = self.walk(items_schema, f)
        return schema

    def handle_frozenset_schema(self, schema: core_schema.FrozenSetSchema, f: Walk) -> core_schema.CoreSchema:
        items_schema = schema.get('items_schema')
        if items_schema is not None:
            schema['items_schema'] = self.walk(items_schema, f)
        return schema

    def handle_generator_schema(self, schema: core_schema.GeneratorSchema, f: Walk) -> core_schema.CoreSchema:
        items_schema = schema.get('items_schema')
        if items_schema is not None:
            schema['items_schema'] = self.walk(items_schema, f)
        return schema

    def handle_tuple_schema(self, schema: core_schema.TupleSchema, f: Walk) -> core_schema.CoreSchema:
        schema['items_schema'] = [self.walk(v, f) for v in schema['items_schema']]
        return schema

    def handle_dict_schema(self, schema: core_schema.DictSchema, f: Walk) -> core_schema.CoreSchema:
        keys_schema = schema.get('keys_schema')
        if keys_schema is not None:
            schema['keys_schema'] = self.walk(keys_schema, f)
        values_schema = schema.get('values_schema')
        if values_schema:
            schema['values_schema'] = self.walk(values_schema, f)
        return schema

    def handle_function_schema(self, schema: AnyFunctionSchema, f: Walk) -> core_schema.CoreSchema:
        if not is_function_with_inner_schema(schema):
            return schema
        schema['schema'] = self.walk(schema['schema'], f)
        return schema

    def handle_union_schema(self, schema: core_schema.UnionSchema, f: Walk) -> core_schema.CoreSchema:
        new_choices: list[CoreSchema | tuple[CoreSchema, str]] = []
        for v in schema['choices']:
            if isinstance(v, tuple):
                new_choices.append((self.walk(v[0], f), v[1]))
            else:
                new_choices.append(self.walk(v, f))
        schema['choices'] = new_choices
        return schema

    def handle_tagged_union_schema(self, schema: core_schema.TaggedUnionSchema, f: Walk) -> core_schema.CoreSchema:
        new_choices: dict[Hashable, core_schema.CoreSchema] = {}
        for k, v in schema['choices'].items():
            new_choices[k] = v if isinstance(v, (str, int)) else self.walk(v, f)
        schema['choices'] = new_choices
        return schema

    def handle_chain_schema(self, schema: core_schema.ChainSchema, f: Walk) -> core_schema.CoreSchema:
        schema['steps'] = [self.walk(v, f) for v in schema['steps']]
        return schema

    def handle_lax_or_strict_schema(self, schema: core_schema.LaxOrStrictSchema, f: Walk) -> core_schema.CoreSchema:
        schema['lax_schema'] = self.walk(schema['lax_schema'], f)
        schema['strict_schema'] = self.walk(schema['strict_schema'], f)
        return schema

    def handle_json_or_python_schema(self, schema: core_schema.JsonOrPythonSchema, f: Walk) -> core_schema.CoreSchema:
        schema['json_schema'] = self.walk(schema['json_schema'], f)
        schema['python_schema'] = self.walk(schema['python_schema'], f)
        return schema

    def handle_model_fields_schema(self, schema: core_schema.ModelFieldsSchema, f: Walk) -> core_schema.CoreSchema:
        extras_schema = schema.get('extras_schema')
        if extras_schema is not None:
            schema['extras_schema'] = self.walk(extras_schema, f)
        replaced_fields: dict[str, core_schema.ModelField] = {}
        replaced_computed_fields: list[core_schema.ComputedField] = []
        for computed_field in schema.get('computed_fields', ()):
            replaced_field = computed_field.copy()
            replaced_field['return_schema'] = self.walk(computed_field['return_schema'], f)
            replaced_computed_fields.append(replaced_field)
        if replaced_computed_fields:
            schema['computed_fields'] = replaced_computed_fields
        for k, v in schema['fields'].items():
            replaced_field = v.copy()
            replaced_field['schema'] = self.walk(v['schema'], f)
            replaced_fields[k] = replaced_field
        schema['fields'] = replaced_fields
        return schema

    def handle_typed_dict_schema(self, schema: core_schema.TypedDictSchema, f: Walk) -> core_schema.CoreSchema:
        extras_schema = schema.get('extras_schema')
        if extras_schema is not None:
            schema['extras_schema'] = self.walk(extras_schema, f)
        replaced_computed_fields: list[core_schema.ComputedField] = []
        for computed_field in schema.get('computed_fields', ()):
            replaced_field = computed_field.copy()
            replaced_field['return_schema'] = self.walk(computed_field['return_schema'], f)
            replaced_computed_fields.append(replaced_field)
        if replaced_computed_fields:
            schema['computed_fields'] = replaced_computed_fields
        replaced_fields: dict[str, core_schema.TypedDictField] = {}
        for k, v in schema['fields'].items():
            replaced_field = v.copy()
            replaced_field['schema'] = self.walk(v['schema'], f)
            replaced_fields[k] = replaced_field
        schema['fields'] = replaced_fields
        return schema

    def handle_dataclass_args_schema(self, schema: core_schema.DataclassArgsSchema, f: Walk) -> core_schema.CoreSchema:
        replaced_fields: list[core_schema.DataclassField] = []
        replaced_computed_fields: list[core_schema.ComputedField] = []
        for computed_field in schema.get('computed_fields', ()):
            replaced_field = computed_field.copy()
            replaced_field['return_schema'] = self.walk(computed_field['return_schema'], f)
            replaced_computed_fields.append(replaced_field)
        if replaced_computed_fields:
            schema['computed_fields'] = replaced_computed_fields
        for field in schema['fields']:
            replaced_field = field.copy()
            replaced_field['schema'] = self.walk(field['schema'], f)
            replaced_fields.append(replaced_field)
        schema['fields'] = replaced_fields
        return schema

    def handle_arguments_schema(self, schema: core_schema.ArgumentsSchema, f: Walk) -> core_schema.CoreSchema:
        replaced_arguments_schema: list[core_schema.ArgumentsParameter] = []
        for param in schema['arguments_schema']:
            replaced_param = param.copy()
            replaced_param['schema'] = self.walk(param['schema'], f)
            replaced_arguments_schema.append(replaced_param)
        schema['arguments_schema'] = replaced_arguments_schema
        if 'var_args_schema' in schema:
            schema['var_args_schema'] = self.walk(schema['var_args_schema'], f)
        if 'var_kwargs_schema' in schema:
            schema['var_kwargs_schema'] = self.walk(schema['var_kwargs_schema'], f)
        return schema

    def handle_call_schema(self, schema: core_schema.CallSchema, f: Walk) -> core_schema.CoreSchema:
        schema['arguments_schema'] = self.walk(schema['arguments_schema'], f)
        if 'return_schema' in schema:
            schema['return_schema'] = self.walk(schema['return_schema'], f)
        return schema