import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
class FSSourceOutputSpec(TraitedSpec):
    T1 = File(exists=True, desc='Intensity normalized whole-head volume', loc='mri')
    aseg = File(exists=True, loc='mri', desc='Volumetric map of regions from automatic segmentation')
    brain = File(exists=True, desc='Intensity normalized brain-only volume', loc='mri')
    brainmask = File(exists=True, desc='Skull-stripped (brain-only) volume', loc='mri')
    filled = File(exists=True, desc='Subcortical mass volume', loc='mri')
    norm = File(exists=True, desc='Normalized skull-stripped volume', loc='mri')
    nu = File(exists=True, desc='Non-uniformity corrected whole-head volume', loc='mri')
    orig = File(exists=True, desc='Base image conformed to Freesurfer space', loc='mri')
    rawavg = File(exists=True, desc='Volume formed by averaging input images', loc='mri')
    ribbon = OutputMultiPath(File(exists=True), desc='Volumetric maps of cortical ribbons', loc='mri', altkey='*ribbon')
    wm = File(exists=True, desc='Segmented white-matter volume', loc='mri')
    wmparc = File(exists=True, loc='mri', desc='Aparc parcellation projected into subcortical white matter')
    curv = OutputMultiPath(File(exists=True), desc='Maps of surface curvature', loc='surf')
    avg_curv = OutputMultiPath(File(exists=True), desc='Average atlas curvature, sampled to subject', loc='surf')
    inflated = OutputMultiPath(File(exists=True), desc='Inflated surface meshes', loc='surf')
    pial = OutputMultiPath(File(exists=True), desc='Gray matter/pia mater surface meshes', loc='surf')
    area_pial = OutputMultiPath(File(exists=True), desc='Mean area of triangles each vertex on the pial surface is associated with', loc='surf', altkey='area.pial')
    curv_pial = OutputMultiPath(File(exists=True), desc='Curvature of pial surface', loc='surf', altkey='curv.pial')
    smoothwm = OutputMultiPath(File(exists=True), loc='surf', desc='Smoothed original surface meshes')
    sphere = OutputMultiPath(File(exists=True), desc='Spherical surface meshes', loc='surf')
    sulc = OutputMultiPath(File(exists=True), desc='Surface maps of sulcal depth', loc='surf')
    thickness = OutputMultiPath(File(exists=True), loc='surf', desc='Surface maps of cortical thickness')
    volume = OutputMultiPath(File(exists=True), desc='Surface maps of cortical volume', loc='surf')
    white = OutputMultiPath(File(exists=True), desc='White/gray matter surface meshes', loc='surf')
    jacobian_white = OutputMultiPath(File(exists=True), desc='Distortion required to register to spherical atlas', loc='surf')
    graymid = OutputMultiPath(File(exists=True), desc='Graymid/midthickness surface meshes', loc='surf', altkey=['graymid', 'midthickness'])
    label = OutputMultiPath(File(exists=True), desc='Volume and surface label files', loc='label', altkey='*label')
    annot = OutputMultiPath(File(exists=True), desc='Surface annotation files', loc='label', altkey='*annot')
    aparc_aseg = OutputMultiPath(File(exists=True), loc='mri', altkey='aparc*aseg', desc='Aparc parcellation projected into aseg volume')
    sphere_reg = OutputMultiPath(File(exists=True), loc='surf', altkey='sphere.reg', desc='Spherical registration file')
    aseg_stats = OutputMultiPath(File(exists=True), loc='stats', altkey='aseg', desc='Automated segmentation statistics file')
    wmparc_stats = OutputMultiPath(File(exists=True), loc='stats', altkey='wmparc', desc='White matter parcellation statistics file')
    aparc_stats = OutputMultiPath(File(exists=True), loc='stats', altkey='aparc', desc='Aparc parcellation statistics files')
    BA_stats = OutputMultiPath(File(exists=True), loc='stats', altkey='BA', desc='Brodmann Area statistics files')
    aparc_a2009s_stats = OutputMultiPath(File(exists=True), loc='stats', altkey='aparc.a2009s', desc='Aparc a2009s parcellation statistics files')
    curv_stats = OutputMultiPath(File(exists=True), loc='stats', altkey='curv', desc='Curvature statistics files')
    entorhinal_exvivo_stats = OutputMultiPath(File(exists=True), loc='stats', altkey='entorhinal_exvivo', desc='Entorhinal exvivo statistics files')