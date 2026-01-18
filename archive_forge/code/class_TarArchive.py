import os
import shutil
import stat
import tarfile
import zipfile
from django.core.exceptions import SuspiciousOperation
class TarArchive(BaseArchive):

    def __init__(self, file):
        self._archive = tarfile.open(file)

    def list(self, *args, **kwargs):
        self._archive.list(*args, **kwargs)

    def extract(self, to_path):
        members = self._archive.getmembers()
        leading = self.has_leading_dir((x.name for x in members))
        for member in members:
            name = member.name
            if leading:
                name = self.split_leading_dir(name)[1]
            filename = self.target_filename(to_path, name)
            if member.isdir():
                if filename:
                    os.makedirs(filename, exist_ok=True)
            else:
                try:
                    extracted = self._archive.extractfile(member)
                except (KeyError, AttributeError) as exc:
                    print('In the tar file %s the member %s is invalid: %s' % (name, member.name, exc))
                else:
                    dirname = os.path.dirname(filename)
                    if dirname:
                        os.makedirs(dirname, exist_ok=True)
                    with open(filename, 'wb') as outfile:
                        shutil.copyfileobj(extracted, outfile)
                        self._copy_permissions(member.mode, filename)
                finally:
                    if extracted:
                        extracted.close()

    def close(self):
        self._archive.close()