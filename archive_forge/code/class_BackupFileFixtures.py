import os
from libcloud.utils.py3 import u
class BackupFileFixtures(FileFixtures):

    def __init__(self, sub_dir=''):
        super().__init__(fixtures_type='backup', sub_dir=sub_dir)