import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
class ClientsideFunction:

    def __init__(self, namespace=None, function_name=None):
        if namespace.startswith('_dashprivate_'):
            raise ValueError("Namespaces cannot start with '_dashprivate_'.")
        if namespace in ['PreventUpdate', 'no_update']:
            raise ValueError(f'"{namespace}" is a forbidden namespace in dash_clientside.')
        self.namespace = namespace
        self.function_name = function_name

    def __repr__(self):
        return f'ClientsideFunction({self.namespace}, {self.function_name})'