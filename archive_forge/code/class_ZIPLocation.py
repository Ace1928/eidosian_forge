import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
class ZIPLocation(Location):
    """Location within a ZIP file.
    """

    def __init__(self, zip, dir):
        """Create a location given an open ZIP file and a path within that
        file.

        :Parameters:
            `zip` : ``zipfile.ZipFile``
                An open ZIP file from the ``zipfile`` module.
            `dir` : str
                A path within that ZIP file.  Can be empty to specify files at
                the top level of the ZIP file.

        """
        self.zip = zip
        self.dir = dir

    def open(self, filename, mode='rb'):
        if self.dir:
            path = self.dir + '/' + filename
        else:
            path = filename
        forward_slash_path = path.replace(os.sep, '/')
        text = self.zip.read(forward_slash_path)
        return BytesIO(text)