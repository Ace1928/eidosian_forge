from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import httpx
import huggingface_hub
import websockets
from packaging import version
from gradio_client import serializing, utils
from gradio_client.exceptions import SerializationSetupError
from gradio_client.utils import (
def _setup_serializers(self) -> tuple[list[serializing.Serializable], list[serializing.Serializable]]:
    inputs = self.dependency['inputs']
    serializers = []
    for i in inputs:
        for component in self.client.config['components']:
            if component['id'] == i:
                component_name = component['type']
                self.input_component_types.append(component_name)
                if component.get('serializer'):
                    serializer_name = component['serializer']
                    if serializer_name not in serializing.SERIALIZER_MAPPING:
                        raise SerializationSetupError(f'Unknown serializer: {serializer_name}, you may need to update your gradio_client version.')
                    serializer = serializing.SERIALIZER_MAPPING[serializer_name]
                elif component_name in serializing.COMPONENT_MAPPING:
                    serializer = serializing.COMPONENT_MAPPING[component_name]
                else:
                    raise SerializationSetupError(f'Unknown component: {component_name}, you may need to update your gradio_client version.')
                serializers.append(serializer())
    outputs = self.dependency['outputs']
    deserializers = []
    for i in outputs:
        for component in self.client.config['components']:
            if component['id'] == i:
                component_name = component['type']
                self.output_component_types.append(component_name)
                if component.get('serializer'):
                    serializer_name = component['serializer']
                    if serializer_name not in serializing.SERIALIZER_MAPPING:
                        raise SerializationSetupError(f'Unknown serializer: {serializer_name}, you may need to update your gradio_client version.')
                    deserializer = serializing.SERIALIZER_MAPPING[serializer_name]
                elif component_name in utils.SKIP_COMPONENTS:
                    deserializer = serializing.SimpleSerializable
                elif component_name in serializing.COMPONENT_MAPPING:
                    deserializer = serializing.COMPONENT_MAPPING[component_name]
                else:
                    raise SerializationSetupError(f'Unknown component: {component_name}, you may need to update your gradio_client version.')
                deserializers.append(deserializer())
    return (serializers, deserializers)