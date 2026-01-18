import os
from libcloud.utils.py3 import u
class DRSFileFixtures(FileFixtures):

    def __init__(self, sub_dir=''):
        super().__init__(fixtures_type='drs', sub_dir=sub_dir)