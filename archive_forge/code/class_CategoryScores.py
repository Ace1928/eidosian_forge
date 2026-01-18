from pydantic import Field as FieldInfo
from .._models import BaseModel
class CategoryScores(BaseModel):
    harassment: float
    "The score for the category 'harassment'."
    harassment_threatening: float = FieldInfo(alias='harassment/threatening')
    "The score for the category 'harassment/threatening'."
    hate: float
    "The score for the category 'hate'."
    hate_threatening: float = FieldInfo(alias='hate/threatening')
    "The score for the category 'hate/threatening'."
    self_harm: float = FieldInfo(alias='self-harm')
    "The score for the category 'self-harm'."
    self_harm_instructions: float = FieldInfo(alias='self-harm/instructions')
    "The score for the category 'self-harm/instructions'."
    self_harm_intent: float = FieldInfo(alias='self-harm/intent')
    "The score for the category 'self-harm/intent'."
    sexual: float
    "The score for the category 'sexual'."
    sexual_minors: float = FieldInfo(alias='sexual/minors')
    "The score for the category 'sexual/minors'."
    violence: float
    "The score for the category 'violence'."
    violence_graphic: float = FieldInfo(alias='violence/graphic')
    "The score for the category 'violence/graphic'."