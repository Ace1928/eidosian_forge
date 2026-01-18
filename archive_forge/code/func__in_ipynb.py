from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
def _in_ipynb():
    """Check if we are running in a Jupyter Notebook.

    Credit to https://stackoverflow.com/a/24937408/3996580
    """
    __VALID_NOTEBOOKS = ["<class 'google.colab._shell.Shell'>", "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"]
    try:
        return str(type(get_ipython())) in __VALID_NOTEBOOKS
    except NameError:
        return False