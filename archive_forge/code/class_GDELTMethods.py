from ._base import *
from .filters import GDELTFilters
from .models import GDELTArticle
class GDELTMethods(Enum):
    article = 'article'
    timeline = 'timeline'