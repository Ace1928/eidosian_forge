import os
import numpy as np
import nibabel as nb
from ..interfaces.base import (
class SignalExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='4-D fMRI nii file')
    label_files = InputMultiPath(File(exists=True), mandatory=True, desc='a 3-D label image, with 0 denoting background, or a list of 3-D probability maps (one per label) or the equivalent 4D file.')
    class_labels = traits.List(mandatory=True, desc='Human-readable labels for each segment in the label file, in order. The length of class_labels must be equal to the number of segments (background excluded). This list corresponds to the class labels in label_file in ascending order')
    out_file = File('signals.tsv', usedefault=True, exists=False, desc='The name of the file to output to. signals.tsv by default')
    incl_shared_variance = traits.Bool(True, usedefault=True, desc='By default (True), returns simple time series calculated from each region independently (e.g., for noise regression). If False, returns unique signals for each region, discarding shared variance (e.g., for connectivity. Only has effect with 4D probability maps.')
    include_global = traits.Bool(False, usedefault=True, desc='If True, include an extra column labeled "GlobalSignal", with values calculated from the entire brain (instead of just regions).')
    detrend = traits.Bool(False, usedefault=True, desc='If True, perform detrending using nilearn.')