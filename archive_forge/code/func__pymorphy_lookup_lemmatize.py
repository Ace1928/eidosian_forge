from typing import Callable, Dict, List, Optional, Tuple
from thinc.api import Model
from ...pipeline import Lemmatizer
from ...pipeline.lemmatizer import lemmatizer_score
from ...symbols import POS
from ...tokens import Token
from ...vocab import Vocab
def _pymorphy_lookup_lemmatize(self, token: Token) -> List[str]:
    string = token.text
    analyses = self._morph.parse(string)
    normal_forms = set([an.normal_form for an in analyses])
    if len(normal_forms) == 1:
        return [next(iter(normal_forms))]
    return [string]