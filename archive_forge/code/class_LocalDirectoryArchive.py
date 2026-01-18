from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import zipfile
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files
from six.moves import urllib
class LocalDirectoryArchive(object):
    """Manages a local zip archive."""
    _ARCHIVE_FILE_NAME = 'apigee_archive_deployment.zip'
    _APIGEE_ARCHIVE_FILE_EXTENSIONS = ['.graphql', '.jar', '.java', '.js', '.jsc', '.json', '.oas', '.properties', '.py', '.securityPolicy', '.wsdl', '.xml', '.xsd', '.xsl', '.yaml', '.yml']
    _ARCHIVE_ROOT = os.path.join('src', 'main', 'apigee')

    def __init__(self, src_dir):
        self._CheckIfPathExists(src_dir)
        if src_dir and (not os.path.isdir(src_dir)):
            raise errors.SourcePathIsNotDirectoryError(src_dir)
        self._src_dir = src_dir if src_dir is not None else files.GetCWD()
        self._tmp_dir = files.TemporaryDirectory()

    def _CheckIfPathExists(self, path):
        """Checks that the given file path exists."""
        if path and (not os.path.exists(path)):
            raise files.MissingFileError('Path to archive deployment does not exist: {}'.format(path))

    def _ZipFileFilter(self, file_name):
        """Filter all files in the archive directory to only allow Apigee files."""
        if not file_name.startswith(self._ARCHIVE_ROOT):
            return False
        _, ext = os.path.splitext(file_name)
        full_path = os.path.join(self._src_dir, file_name)
        if os.path.basename(full_path).startswith('.'):
            return False
        if os.path.isdir(full_path):
            return True
        if os.path.isfile(full_path) and ext.lower() in self._APIGEE_ARCHIVE_FILE_EXTENSIONS:
            return True
        return False

    def Zip(self):
        """Creates a zip archive of the specified directory."""
        dst_file = os.path.join(self._tmp_dir.path, self._ARCHIVE_FILE_NAME)
        archive.MakeZipFromDir(dst_file, self._src_dir, self._ZipFileFilter)
        return dst_file

    def ValidateZipFilePath(self, zip_path):
        """Checks that the zip file path exists and the file is a zip archvie."""
        self._CheckIfPathExists(zip_path)
        if not zipfile.is_zipfile(zip_path):
            raise errors.BundleFileNotValidError(zip_path)

    def Close(self):
        """Deletes the local temporary directory."""
        return self._tmp_dir.Close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, val, tb):
        try:
            self.Close()
        except:
            log.warning('Temporary directory was not successfully deleted.')
            return True