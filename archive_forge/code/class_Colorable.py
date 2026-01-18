import pygments
from traitlets.config import Configurable
from traitlets import Unicode
class Colorable(Configurable):
    """
    A subclass of configurable for all the classes that have a `default_scheme`
    """
    default_style = Unicode('LightBG').tag(config=True)