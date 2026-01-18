from __future__ import annotations
from typing import Final
import streamlit as st
from streamlit import config
from streamlit.errors import UncaughtAppException
from streamlit.logger import get_logger
class ConfigurablePanel(panel.Panel):

    def __init__(self, renderable, box=box.Box('────\n    \n────\n    \n────\n────\n    \n────\n'), **kwargs):
        super().__init__(renderable, box, **kwargs)