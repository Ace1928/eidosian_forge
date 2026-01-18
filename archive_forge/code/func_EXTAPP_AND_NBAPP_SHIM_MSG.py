from functools import wraps
from copy import deepcopy
from traitlets import TraitError
from traitlets.config.loader import (
from jupyter_core.application import JupyterApp
from jupyter_server.serverapp import ServerApp
from jupyter_server.extension.application import ExtensionApp
from .traits import NotebookAppTraits
def EXTAPP_AND_NBAPP_SHIM_MSG(trait_name, extapp_name):
    return "'{trait_name}' is found in both {extapp_name} and NotebookApp. This is a recent change. This config will only be set in {extapp_name}. Please check if you should also config these traits in NotebookApp for your purpose.".format(trait_name=trait_name, extapp_name=extapp_name)