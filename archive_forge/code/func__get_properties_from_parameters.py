from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def _get_properties_from_parameters(parameters: List[Parameter], spec: OpenAPISpec) -> List[APIProperty]:
    """Get the properties of the operation."""
    properties = []
    for param in parameters:
        if APIProperty.is_supported_location(param.param_in):
            properties.append(APIProperty.from_parameter(param, spec))
        elif param.required:
            raise ValueError(INVALID_LOCATION_TEMPL.format(location=param.param_in, name=param.name))
        else:
            logger.warning(INVALID_LOCATION_TEMPL.format(location=param.param_in, name=param.name) + ' Ignoring optional parameter')
            pass
    return properties