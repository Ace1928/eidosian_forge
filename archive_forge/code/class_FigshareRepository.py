import os
import sys
import ftplib
import warnings
from .utils import parse_url
class FigshareRepository(DataRepository):

    def __init__(self, doi, archive_url):
        self.archive_url = archive_url
        self.doi = doi
        self._api_response = None

    @classmethod
    def initialize(cls, doi, archive_url):
        """
        Initialize the data repository if the given URL points to a
        corresponding repository.

        Initializes a data repository object. This is done as part of
        a chain of responsibility. If the class cannot handle the given
        repository URL, it returns `None`. Otherwise a `DataRepository`
        instance is returned.

        Parameters
        ----------
        doi : str
            The DOI that identifies the repository
        archive_url : str
            The resolved URL for the DOI
        """
        parsed_archive_url = parse_url(archive_url)
        if parsed_archive_url['netloc'] != 'figshare.com':
            return None
        return cls(doi, archive_url)

    def _parse_version_from_doi(self):
        """
        Parse version from the doi

        Return None if version is not available in the doi.
        """
        _, suffix = self.doi.split('/')
        last_part = suffix.split('.')[-1]
        if last_part[0] != 'v':
            return None
        version = int(last_part[1:])
        return version

    @property
    def api_response(self):
        """Cached API response from Figshare"""
        if self._api_response is None:
            import requests
            article = requests.get(f'https://api.figshare.com/v2/articles?doi={self.doi}', timeout=5).json()[0]
            article_id = article['id']
            version = self._parse_version_from_doi()
            if version is None:
                warnings.warn(f"The Figshare DOI '{self.doi}' doesn't specify which version of the repository should be used. Figshare will point to the latest version available.", UserWarning)
                api_url = f'https://api.figshare.com/v2/articles/{article_id}'
            else:
                api_url = f'https://api.figshare.com/v2/articles/{article_id}/versions/{version}'
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            self._api_response = response.json()['files']
        return self._api_response

    def download_url(self, file_name):
        """
        Use the repository API to get the download URL for a file given
        the archive URL.

        Parameters
        ----------
        file_name : str
            The name of the file in the archive that will be downloaded.

        Returns
        -------
        download_url : str
            The HTTP URL that can be used to download the file.
        """
        files = {item['name']: item for item in self.api_response}
        if file_name not in files:
            raise ValueError(f"File '{file_name}' not found in data archive {self.archive_url} (doi:{self.doi}).")
        download_url = files[file_name]['download_url']
        return download_url

    def populate_registry(self, pooch):
        """
        Populate the registry using the data repository's API

        Parameters
        ----------
        pooch : Pooch
            The pooch instance that the registry will be added to.
        """
        for filedata in self.api_response:
            pooch.registry[filedata['name']] = f'md5:{filedata['computed_md5']}'