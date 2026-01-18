import os
from abc import abstractmethod
from ... import logging
from ..base import File, BaseInterface, BaseInterfaceInputSpec, TraitedSpec
@abstractmethod
def _generate_report(self):
    """
        Saves report to file identified by _out_report instance variable
        """
    raise NotImplementedError