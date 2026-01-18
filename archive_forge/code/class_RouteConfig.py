import json
import logging
import os
import pathlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pydantic
import yaml
from packaging import version
from packaging.version import Version
from pydantic import ConfigDict, Field, ValidationError, root_validator, validator
from pydantic.json import pydantic_encoder
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel, LimitModel, ResponseModel
from mlflow.gateway.constants import (
from mlflow.gateway.utils import (
class RouteConfig(AliasedConfigModel):
    name: str
    route_type: RouteType = Field(alias='endpoint_type')
    model: Model
    limit: Optional[Limit] = None

    @validator('name')
    def validate_endpoint_name(cls, route_name):
        if not is_valid_endpoint_name(route_name):
            raise MlflowException.invalid_parameter_value(f"The route name provided contains disallowed characters for a url endpoint. '{route_name}' is invalid. Names cannot contain spaces or any non alphanumeric characters other than hyphen and underscore.")
        return route_name

    @validator('model', pre=True)
    def validate_model(cls, model):
        if model:
            model_instance = Model(**model)
            if model_instance.provider in Provider.values() and model_instance.config is None:
                raise MlflowException.invalid_parameter_value(f'A config must be supplied when setting a provider. The provider entry for {model_instance.provider} is incorrect.')
        return model

    @root_validator(skip_on_failure=True)
    def validate_route_type_and_model_name(cls, values):
        route_type = values.get('route_type')
        model = values.get('model')
        if model and model.provider == 'mosaicml' and (route_type == RouteType.LLM_V1_CHAT) and (not is_valid_mosiacml_chat_model(model.name)):
            raise MlflowException.invalid_parameter_value(f"An invalid model has been specified for the chat route. '{model.name}'. Ensure the model selected starts with one of: {MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES}")
        if model and model.provider == 'ai21labs' and (not is_valid_ai21labs_model(model.name)):
            raise MlflowException.invalid_parameter_value(f"An Unsupported AI21Labs model has been specified: '{model.name}'. Please see documentation for supported models.")
        return values

    @validator('route_type', pre=True)
    def validate_route_type(cls, value):
        if value in RouteType._value2member_map_:
            return value
        raise MlflowException.invalid_parameter_value(f"The route_type '{value}' is not supported.")

    @validator('limit', pre=True)
    def validate_limit(cls, value):
        from limits import parse
        if value:
            limit = Limit(**value)
            try:
                parse(f'{limit.calls}/{limit.renewal_period}')
            except ValueError:
                raise MlflowException.invalid_parameter_value('Failed to parse the rate limit configuration.Please make sure limit.calls is a positive number andlimit.renewal_period is a right granularity')
        return value

    def to_route(self) -> 'Route':
        return Route(name=self.name, route_type=self.route_type, model=RouteModelInfo(name=self.model.name, provider=self.model.provider), route_url=f'{MLFLOW_GATEWAY_ROUTE_BASE}{self.name}{MLFLOW_QUERY_SUFFIX}', limit=self.limit)