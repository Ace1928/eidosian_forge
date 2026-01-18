from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def _serialize_io(run_inputs: Optional[dict]) -> dict:
    if not run_inputs:
        return {}
    from google.protobuf.json_format import MessageToJson
    from google.protobuf.message import Message
    serialized_inputs = {}
    for key, value in run_inputs.items():
        if isinstance(value, Message):
            serialized_inputs[key] = MessageToJson(value)
        elif key == 'input_documents':
            serialized_inputs.update({f'input_document_{i}': doc.json() for i, doc in enumerate(value)})
        else:
            serialized_inputs[key] = value
    return serialized_inputs