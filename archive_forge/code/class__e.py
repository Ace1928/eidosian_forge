from os_ken.lib import stringify
from lxml import objectify
import lxml.etree as ET
class _e(object):

    def __init__(self, name, is_list):
        self.name = name
        self.cls = None
        self.is_list = is_list