from __future__ import annotations
import inspect
from typing import (
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
    config = super()._merge_configs(*configs)
    expected_keys = [field_spec.id for field_spec in self.history_factory_config]
    configurable = config.get('configurable', {})
    missing_keys = set(expected_keys) - set(configurable.keys())
    if missing_keys:
        example_input = {self.input_messages_key: 'foo'}
        example_configurable = {missing_key: '[your-value-here]' for missing_key in missing_keys}
        example_config = {'configurable': example_configurable}
        raise ValueError(f"Missing keys {sorted(missing_keys)} in config['configurable'] Expected keys are {sorted(expected_keys)}.When using via .invoke() or .stream(), pass in a config; e.g., chain.invoke({example_input}, {example_config})")
    parameter_names = _get_parameter_names(self.get_session_history)
    if len(expected_keys) == 1:
        message_history = self.get_session_history(configurable[expected_keys[0]])
    else:
        if set(expected_keys) != set(parameter_names):
            raise ValueError(f'Expected keys {sorted(expected_keys)} do not match parameter names {sorted(parameter_names)} of get_session_history.')
        message_history = self.get_session_history(**{key: configurable[key] for key in expected_keys})
    config['configurable']['message_history'] = message_history
    return config