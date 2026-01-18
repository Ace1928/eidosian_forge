import os
import warnings
import xml.dom.minidom
from .base import (
class SlicerCommandLineInputSpec(DynamicTraitedSpec, CommandLineInputSpec):
    module = traits.Str(desc='name of the Slicer command line module you want to use')