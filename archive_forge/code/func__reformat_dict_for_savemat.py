import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
def _reformat_dict_for_savemat(self, contents):
    """Encloses a dict representation within hierarchical lists.

        In order to create an appropriate SPM job structure, a Python
        dict storing the job needs to be modified so that each dict
        embedded in dict needs to be enclosed as a list element.

        Examples
        --------
        >>> a = SPMCommand()._reformat_dict_for_savemat(dict(a=1,
        ...                                                  b=dict(c=2, d=3)))
        >>> a == [{'a': 1, 'b': [{'c': 2, 'd': 3}]}]
        True

        """
    newdict = {}
    try:
        for key, value in list(contents.items()):
            if isinstance(value, dict):
                if value:
                    newdict[key] = self._reformat_dict_for_savemat(value)
            else:
                newdict[key] = value
        return [newdict]
    except TypeError:
        print('Requires dict input')