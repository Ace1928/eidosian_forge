import abc
import logging
from cliff import command
from cliff import lister
from cliff import show
from osc_lib import exceptions
from osc_lib.i18n import _
class ShowOne(Command, show.ShowOne):
    pass