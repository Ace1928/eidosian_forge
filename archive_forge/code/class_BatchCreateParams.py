from __future__ import annotations
from typing import Dict, Optional
from typing_extensions import Literal, Required, TypedDict
class BatchCreateParams(TypedDict, total=False):
    completion_window: Required[Literal['24h']]
    'The time frame within which the batch should be processed.\n\n    Currently only `24h` is supported.\n    '
    endpoint: Required[Literal['/v1/chat/completions']]
    'The endpoint to be used for all requests in the batch.\n\n    Currently only `/v1/chat/completions` is supported.\n    '
    input_file_id: Required[str]
    'The ID of an uploaded file that contains requests for the new batch.\n\n    See [upload file](https://platform.openai.com/docs/api-reference/files/create)\n    for how to upload a file.\n\n    Your input file must be formatted as a JSONL file, and must be uploaded with the\n    purpose `batch`.\n    '
    metadata: Optional[Dict[str, str]]
    'Optional custom metadata for the batch.'