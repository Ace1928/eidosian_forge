import os
import zipfile
import shutil
import numpy as np
from . import _constants as C
from ...data import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from ....contrib import text
from .... import nd, base
def _build_vocab(self, content):
    if not self._counter:
        self._counter = text.utils.count_tokens_from_str(content)
    if not self._vocab:
        self._vocab = text.vocab.Vocabulary(counter=self.frequencies, reserved_tokens=[C.EOS_TOKEN])