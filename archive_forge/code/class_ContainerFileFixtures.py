import os
from libcloud.utils.py3 import u
class ContainerFileFixtures(FileFixtures):

    def __init__(self, sub_dir=''):
        super().__init__(fixtures_type='container', sub_dir=sub_dir)