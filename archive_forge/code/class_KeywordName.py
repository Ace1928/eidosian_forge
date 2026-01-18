import pydoc
from contextlib import suppress
from typing import Dict, Optional
from jedi.inference.names import AbstractArbitraryName
class KeywordName(AbstractArbitraryName):
    api_type = 'keyword'

    def py__doc__(self):
        return imitate_pydoc(self.string_name)