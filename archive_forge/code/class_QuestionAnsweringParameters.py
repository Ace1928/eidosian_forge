from dataclasses import dataclass
from typing import Optional
from .base import BaseInferenceType
@dataclass
class QuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Question Answering
    """
    align_to_words: Optional[bool] = None
    'Attempts to align the answer to real words. Improves quality on space separated\n    languages. Might hurt on non-space-separated languages (like Japanese or Chinese)\n    '
    doc_stride: Optional[int] = None
    'If the context is too long to fit with the question for the model, it will be split in\n    several chunks with some overlap. This argument controls the size of that overlap.\n    '
    handle_impossible_answer: Optional[bool] = None
    'Whether to accept impossible as an answer.'
    max_answer_len: Optional[int] = None
    'The maximum length of predicted answers (e.g., only answers with a shorter length are\n    considered).\n    '
    max_question_len: Optional[int] = None
    'The maximum length of the question after tokenization. It will be truncated if needed.'
    max_seq_len: Optional[int] = None
    'The maximum length of the total sentence (context + question) in tokens of each chunk\n    passed to the model. The context will be split in several chunks (using docStride as\n    overlap) if needed.\n    '
    top_k: Optional[int] = None
    'The number of answers to return (will be chosen by order of likelihood). Note that we\n    return less than topk answers if there are not enough options available within the\n    context.\n    '