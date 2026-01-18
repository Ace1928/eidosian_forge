import os
from abc import abstractmethod
from ... import logging
from ..base import File, BaseInterface, BaseInterfaceInputSpec, TraitedSpec
class ReportCapableInputSpec(BaseInterfaceInputSpec):
    out_report = File('report', usedefault=True, hash_files=False, desc='filename for the visual report')