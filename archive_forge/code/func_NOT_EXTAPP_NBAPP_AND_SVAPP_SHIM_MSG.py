from functools import wraps
from copy import deepcopy
from traitlets import TraitError
from traitlets.config.loader import (
from jupyter_core.application import JupyterApp
from jupyter_server.serverapp import ServerApp
from jupyter_server.extension.application import ExtensionApp
from .traits import NotebookAppTraits
def NOT_EXTAPP_NBAPP_AND_SVAPP_SHIM_MSG(trait_name, extapp_name):
    return "'{trait_name}' is not found in {extapp_name}, but it was found in both NotebookApp and ServerApp. This is likely a recent change. This config will only be set in ServerApp. Please check if you should also config these traits in NotebookApp for your purpose.".format(trait_name=trait_name, extapp_name=extapp_name)