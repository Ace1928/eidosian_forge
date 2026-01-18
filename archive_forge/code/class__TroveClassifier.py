import logging
import os
import re
import string
import typing
from itertools import chain as _chain
class _TroveClassifier:
    """The ``trove_classifiers`` package is the official way of validating classifiers,
    however this package might not be always available.
    As a workaround we can still download a list from PyPI.
    We also don't want to be over strict about it, so simply skipping silently is an
    option (classifiers will be validated anyway during the upload to PyPI).
    """
    downloaded: typing.Union[None, 'Literal[False]', typing.Set[str]]

    def __init__(self):
        self.downloaded = None
        self._skip_download = False
        self.__name__ = 'trove_classifier'

    def _disable_download(self):
        self._skip_download = True

    def __call__(self, value: str) -> bool:
        if self.downloaded is False or self._skip_download is True:
            return True
        if os.getenv('NO_NETWORK') or os.getenv('VALIDATE_PYPROJECT_NO_NETWORK'):
            self.downloaded = False
            msg = 'Install ``trove-classifiers`` to ensure proper validation. Skipping download of classifiers list from PyPI (NO_NETWORK).'
            _logger.debug(msg)
            return True
        if self.downloaded is None:
            msg = 'Install ``trove-classifiers`` to ensure proper validation. Meanwhile a list of classifiers will be downloaded from PyPI.'
            _logger.debug(msg)
            try:
                self.downloaded = set(_download_classifiers().splitlines())
            except Exception:
                self.downloaded = False
                _logger.debug('Problem with download, skipping validation')
                return True
        return value in self.downloaded or value.lower().startswith('private ::')