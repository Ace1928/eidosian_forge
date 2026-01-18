from typing import TYPE_CHECKING, List, TypedDict
class QuestionAnsweringOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.question_answering`] task.

    Args:
        score (`float`):
            A float that represents how likely that the answer is correct.
        start (`int`):
            The index (string wise) of the start of the answer within context.
        end (`int`):
            The index (string wise) of the end of the answer within context.
        answer (`str`):
            A string that is the answer within the text.
    """
    score: float
    start: int
    end: int
    answer: str