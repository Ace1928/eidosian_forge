import json
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.cross_encoders.base import BaseCrossEncoder
class CrossEncoderContentHandler:
    """Content handler for CrossEncoder class."""
    content_type = 'application/json'
    accepts = 'application/json'

    def transform_input(self, text_pairs: List[Tuple[str, str]]) -> bytes:
        input_str = json.dumps({'text_pairs': text_pairs})
        return input_str.encode('utf-8')

    def transform_output(self, output: Any) -> List[float]:
        response_json = json.loads(output.read().decode('utf-8'))
        scores = response_json['scores']
        return scores