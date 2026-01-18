import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Union
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
class CMUDict(Dataset):
    """*CMU Pronouncing Dictionary* :cite:`cmudict` (CMUDict) dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        exclude_punctuations (bool, optional):
            When enabled, exclude the pronounciation of punctuations, such as
            `!EXCLAMATION-POINT` and `#HASH-MARK`.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str, optional):
            The URL to download the dictionary from.
            (default: ``"http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"``)
        url_symbols (str, optional):
            The URL to download the list of symbols from.
            (default: ``"http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols"``)
    """

    def __init__(self, root: Union[str, Path], exclude_punctuations: bool=True, *, download: bool=False, url: str='http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b', url_symbols: str='http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols') -> None:
        self.exclude_punctuations = exclude_punctuations
        self._root_path = Path(root)
        if not os.path.isdir(self._root_path):
            raise RuntimeError(f'The root directory does not exist; {root}')
        dict_file = self._root_path / os.path.basename(url)
        symbol_file = self._root_path / os.path.basename(url_symbols)
        if not os.path.exists(dict_file):
            if not download:
                raise RuntimeError(f'The dictionary file is not found in the following location. Set `download=True` to download it. {dict_file}')
            checksum = _CHECKSUMS.get(url, None)
            download_url_to_file(url, dict_file, checksum)
        if not os.path.exists(symbol_file):
            if not download:
                raise RuntimeError(f'The symbol file is not found in the following location. Set `download=True` to download it. {symbol_file}')
            checksum = _CHECKSUMS.get(url_symbols, None)
            download_url_to_file(url_symbols, symbol_file, checksum)
        with open(symbol_file, 'r') as text:
            self._symbols = [line.strip() for line in text.readlines()]
        with open(dict_file, 'r', encoding='latin-1') as text:
            self._dictionary = _parse_dictionary(text.readlines(), exclude_punctuations=self.exclude_punctuations)

    def __getitem__(self, n: int) -> Tuple[str, List[str]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            Tuple of a word and its phonemes

            str:
                Word
            List[str]:
                Phonemes
        """
        return self._dictionary[n]

    def __len__(self) -> int:
        return len(self._dictionary)

    @property
    def symbols(self) -> List[str]:
        """list[str]: A list of phonemes symbols, such as ``"AA"``, ``"AE"``, ``"AH"``."""
        return self._symbols.copy()