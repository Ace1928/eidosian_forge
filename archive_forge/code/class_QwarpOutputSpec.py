import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class QwarpOutputSpec(TraitedSpec):
    warped_source = File(desc='Warped source file. If plusminus is used, this is the undistortedsource file.')
    warped_base = File(desc='Undistorted base file.')
    source_warp = File(desc="Displacement in mm for the source image.If plusminus is used this is the field suceptibility correctionwarp (in 'mm') for source image.")
    base_warp = File(desc="Displacement in mm for the base image.If plus minus is used, this is the field suceptibility correctionwarp (in 'mm') for base image. This is only output if plusminusor iwarp options are passed")
    weights = File(desc='Auto-computed weight volume.')