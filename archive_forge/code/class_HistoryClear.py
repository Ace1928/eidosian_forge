import sqlite3
from pathlib import Path
from traitlets.config.application import Application
from .application import BaseIPythonApplication
from traitlets import Bool, Int, Dict
from ..utils.io import ask_yes_no
class HistoryClear(HistoryTrim):
    description = clear_hist_help
    keep = Int(0, help='Number of recent lines to keep in the database.')
    force = Bool(False, help="Don't prompt user for confirmation").tag(config=True)
    flags = Dict(dict(force=({'HistoryClear': {'force': True}}, force.help), f=({'HistoryTrim': {'force': True}}, force.help)))
    aliases = Dict()

    def start(self):
        if self.force or ask_yes_no('Really delete all ipython history? ', default='no', interrupt='no'):
            HistoryTrim.start(self)