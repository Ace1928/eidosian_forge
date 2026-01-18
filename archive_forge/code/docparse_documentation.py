from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
Replace flags with parameter names.

    This is a simple operation where we replace the command line flags
    with the attribute names.

    Parameters
    ----------
    rep_doc : string
        Documentation string
    opts : dict
        Dictionary of option attributes and keys.  Use reverse_opt_map
        to reverse flags and attrs from opt_map class attribute.

    Returns
    -------
    rep_doc : string
        New docstring with flags replaces with attribute names.

    Examples
    --------
    doc = grab_doc('bet')
    opts = reverse_opt_map(fsl.Bet.opt_map)
    rep_doc = replace_opts(doc, opts)

    