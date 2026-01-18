import asyncio
import copy
import io
import os
import sys
import IPython
import traitlets
import ipyvuetify as v
@traitlets.observe('file_info')
def _file_info_changed(self, _):
    self.version += 1
    self.reset_stats()