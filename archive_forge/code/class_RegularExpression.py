import re
import click
import json
from .instance import import_module
from ..interfaces.base import InputMultiPath, traits
from ..interfaces.base.support import get_trait_desc
class RegularExpression(click.ParamType):
    name = 'regex'

    def convert(self, value, param, ctx):
        try:
            rex = re.compile(value, re.IGNORECASE)
        except ValueError:
            self.fail('%s is not a valid regular expression.' % value, param, ctx)
        else:
            return rex