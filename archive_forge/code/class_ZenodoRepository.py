import os
import sys
import ftplib
import warnings
from .utils import parse_url
class ZenodoRepository(DataRepository):
    base_api_url = 'https://zenodo.org/api/records'

    def __init__(self, doi, archive_url):
        self.archive_url = archive_url
        self.doi = doi
        self._api_response = None
        self._api_version = None

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
        if parsed_archive_url['netloc'] != 'zenodo.org':
            return None
        return cls(doi, archive_url)

    @property
    def api_response(self):
        """Cached API response from Zenodo"""
        if self._api_response is None:
            import requests
            article_id = self.archive_url.split('/')[-1]
            self._api_response = requests.get(f'{self.base_api_url}/{article_id}', timeout=5).json()
        return self._api_response

    @property
    def api_version(self):
        """
        Version of the Zenodo API we are interacting with

        The versions can either be :

        - ``"legacy"``: corresponds to the Zenodo API that was supported until
          2023-10-12 (before the migration to InvenioRDM).
        - ``"new"``: corresponds to the new API that went online on 2023-10-13
          after the migration to InvenioRDM.

        The ``"new"`` API breaks backward compatibility with the ``"legacy"``
        one and could probably be replaced by an updated version that restores
        the behaviour of the ``"legacy"`` one.

        Returns
        -------
        str
        """
        if self._api_version is None:
            if all(('key' in file for file in self.api_response['files'])):
                self._api_version = 'legacy'
            elif all(('filename' in file for file in self.api_response['files'])):
                self._api_version = 'new'
            else:
                raise ValueError(f"Couldn't determine the version of the Zenodo API for {self.archive_url} (doi:{self.doi}).")
        return self._api_version

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

        Notes
        -----
        After Zenodo migrated to InvenioRDM on Oct 2023, their API changed. The
        link to the desired files that appears in the API response leads to 404
        errors (by 2023-10-17). The files are available in the following url:
        ``https://zenodo.org/records/{article_id}/files/{file_name}?download=1``.

        This method supports both the legacy and the new API.
        """
        if self.api_version == 'legacy':
            files = {item['key']: item for item in self.api_response['files']}
        else:
            files = [item['filename'] for item in self.api_response['files']]
        if file_name not in files:
            raise ValueError(f"File '{file_name}' not found in data archive {self.archive_url} (doi:{self.doi}).")
        if self.api_version == 'legacy':
            download_url = files[file_name]['links']['self']
        else:
            article_id = self.api_response['id']
            download_url = f'https://zenodo.org/records/{article_id}/files/{file_name}?download=1'
        return download_url

    def populate_registry(self, pooch):
        """
        Populate the registry using the data repository's API

        Parameters
        ----------
        pooch : Pooch
            The pooch instance that the registry will be added to.

        Notes
        -----
        After Zenodo migrated to InvenioRDM on Oct 2023, their API changed. The
        checksums for each file listed in the API reference is now an md5 sum.

        This method supports both the legacy and the new API.
        """
        for filedata in self.api_response['files']:
            checksum = filedata['checksum']
            if self.api_version == 'legacy':
                key = 'key'
            else:
                key = 'filename'
                checksum = f'md5:{checksum}'
            pooch.registry[filedata[key]] = checksum