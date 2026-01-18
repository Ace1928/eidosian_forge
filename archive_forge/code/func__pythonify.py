from os_ken.lib import stringify
from lxml import objectify
import lxml.etree as ET
def _pythonify(name):
    return name.replace('-', '_')