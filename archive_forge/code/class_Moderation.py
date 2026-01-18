from pydantic import Field as FieldInfo
from .._models import BaseModel
class Moderation(BaseModel):
    categories: Categories
    'A list of the categories, and whether they are flagged or not.'
    category_scores: CategoryScores
    'A list of the categories along with their scores as predicted by model.'
    flagged: bool
    'Whether any of the below categories are flagged.'