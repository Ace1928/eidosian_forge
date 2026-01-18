import warnings
import snowballstemmer
from sphinx.deprecation import RemovedInSphinx70Warning
class BaseStemmer:

    def __init__(self) -> None:
        warnings.warn(f"{self.__class__.__name__} is deprecated, use snowballstemmer.stemmer('porter') instead.", RemovedInSphinx70Warning, stacklevel=3)

    def stem(self, word: str) -> str:
        raise NotImplementedError