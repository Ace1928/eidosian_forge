import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
class _Wildcard:

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def __repr__(self):
        return f'<{self}>'

    def to_json(self):
        return f'["{self._name}"]'