import pytest
from packaging.version import Version
from collections import namedtuple
from ...base import traits, File, TraitedSpec, BaseInterfaceInputSpec
from ..base import (
class DummyWorkflow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'dwf1'

    def run(self, in_files, param1=1, out_dir='', out_ref='out1.txt'):
        """Workflow used to test basic workflows.

            Parameters
            ----------
            in_files : string
                fake input string param
            param1 : int, optional
                fake positional param (default 1)
            out_dir : string, optional
                fake output directory (default '')
            out_ref : string, optional
                fake out file (default out1.txt)

            References
            -----------
            dummy references

            """
        return param1