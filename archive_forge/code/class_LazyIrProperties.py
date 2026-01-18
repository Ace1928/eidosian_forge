from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
class LazyIrProperties:
    """Collection of properties for an IR node

    The property groups are listed below. Each group is mutually
    exclusive, meaning that only one property from each group can be True
    at any one time. The properties can be accessed as if they were normal
    attributes. The mutual exclusivity is automatically handled.
    """
    Properties: Tuple[Tuple[str, ...], ...] = (('ShapePrecompute', 'ShapeCompute', 'ShapeCache'), ('Lower', 'LowerDeclOnly'), ('CanBeReused', 'CanBeReusedDeclOnly'), ('CreateFn', 'CreateFnDeclOnly'), ('TreatScalarsAsConstants',))

    def __init__(self, *default_properties: str):
        properties: Dict[Tuple[str, ...], Optional[str]] = {p: None for p in LazyIrProperties.Properties}
        self.__dict__['properties'] = properties
        for p in default_properties:
            setattr(self, p, True)

    def __getattr__(self, key: str) -> Any:
        properties = self.__dict__['properties']
        for values in LazyIrProperties.Properties:
            if key in values:
                return properties[values] == key
        return self.__getattribute__(key)

    def __setattr__(self, key: str, value: Any) -> Any:
        properties = self.__dict__['properties']
        for values in LazyIrProperties.Properties:
            if key in values:
                properties[values] = key if value else None
                return value
        raise KeyError(f'Invalid property: {key}')