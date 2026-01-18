import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
class ProviderDesc:
    """Simple class describing an Enchant provider.

    Each provider has the following information associated with it:

        * name:        Internal provider name (e.g. "aspell")
        * desc:        Human-readable description (e.g. "Aspell Provider")
        * file:        Location of the library containing the provider

    """
    _DOC_ERRORS = ['desc']

    def __init__(self, name, desc, file):
        self.name = name
        self.desc = desc
        self.file = file

    def __str__(self):
        return '<Enchant: %s>' % self.desc

    def __repr__(self):
        return str(self)

    def __eq__(self, pd):
        """Equality operator on ProviderDesc objects."""
        return self.name == pd.name and self.desc == pd.desc and (self.file == pd.file)

    def __hash__(self):
        """Hash operator on ProviderDesc objects."""
        return hash(self.name + self.desc + self.file)