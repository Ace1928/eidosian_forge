import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MELODICInputSpec(FSLCommandInputSpec):
    in_files = InputMultiPath(File(exists=True), argstr='-i %s', mandatory=True, position=0, desc='input file names (either single file name or a list)', sep=',')
    out_dir = Directory(argstr='-o %s', desc='output directory name', genfile=True)
    mask = File(exists=True, argstr='-m %s', desc='file name of mask for thresholding')
    no_mask = traits.Bool(argstr='--nomask', desc='switch off masking')
    update_mask = traits.Bool(argstr='--update_mask', desc='switch off mask updating')
    no_bet = traits.Bool(argstr='--nobet', desc='switch off BET')
    bg_threshold = traits.Float(argstr='--bgthreshold=%f', desc='brain/non-brain threshold used to mask non-brain voxels, as a percentage (only if --nobet selected)')
    dim = traits.Int(argstr='-d %d', desc='dimensionality reduction into #num dimensions (default: automatic estimation)')
    dim_est = traits.Str(argstr='--dimest=%s', desc='use specific dim. estimation technique: lap, bic, mdl, aic, mean (default: lap)')
    sep_whiten = traits.Bool(argstr='--sep_whiten', desc='switch on separate whitening')
    sep_vn = traits.Bool(argstr='--sep_vn', desc='switch off joined variance normalization')
    migp = traits.Bool(argstr='--migp', desc='switch on MIGP data reduction')
    migpN = traits.Int(argstr='--migpN %d', desc='number of internal Eigenmaps')
    migp_shuffle = traits.Bool(argstr='--migp_shuffle', desc='randomise MIGP file order (default: TRUE)')
    migp_factor = traits.Int(argstr='--migp_factor %d', desc='Internal Factor of mem-threshold relative to number of Eigenmaps (default: 2)')
    num_ICs = traits.Int(argstr='-n %d', desc="number of IC's to extract (for deflation approach)")
    approach = traits.Str(argstr='-a %s', desc='approach for decomposition, 2D: defl, symm (default), 3D: tica (default), concat')
    non_linearity = traits.Str(argstr='--nl=%s', desc='nonlinearity: gauss, tanh, pow3, pow4')
    var_norm = traits.Bool(argstr='--vn', desc='switch off variance normalization')
    pbsc = traits.Bool(argstr='--pbsc', desc='switch off conversion to percent BOLD signal change')
    cov_weight = traits.Float(argstr='--covarweight=%f', desc='voxel-wise weights for the covariance matrix (e.g. segmentation information)')
    epsilon = traits.Float(argstr='--eps=%f', desc='minimum error change')
    epsilonS = traits.Float(argstr='--epsS=%f', desc='minimum error change for rank-1 approximation in TICA')
    maxit = traits.Int(argstr='--maxit=%d', desc='maximum number of iterations before restart')
    max_restart = traits.Int(argstr='--maxrestart=%d', desc='maximum number of restarts')
    mm_thresh = traits.Float(argstr='--mmthresh=%f', desc='threshold for Mixture Model based inference')
    no_mm = traits.Bool(argstr='--no_mm', desc='switch off mixture modelling on IC maps')
    ICs = File(exists=True, argstr='--ICs=%s', desc='filename of the IC components file for mixture modelling')
    mix = File(exists=True, argstr='--mix=%s', desc='mixing matrix for mixture modelling / filtering')
    smode = File(exists=True, argstr='--smode=%s', desc='matrix of session modes for report generation')
    rem_cmp = traits.List(traits.Int, argstr='-f %d', desc='component numbers to remove')
    report = traits.Bool(argstr='--report', desc='generate Melodic web report')
    bg_image = File(exists=True, argstr='--bgimage=%s', desc='specify background image for report (default: mean image)')
    tr_sec = traits.Float(argstr='--tr=%f', desc='TR in seconds')
    log_power = traits.Bool(argstr='--logPower', desc='calculate log of power for frequency spectrum')
    t_des = File(exists=True, argstr='--Tdes=%s', desc='design matrix across time-domain')
    t_con = File(exists=True, argstr='--Tcon=%s', desc='t-contrast matrix across time-domain')
    s_des = File(exists=True, argstr='--Sdes=%s', desc='design matrix across subject-domain')
    s_con = File(exists=True, argstr='--Scon=%s', desc='t-contrast matrix across subject-domain')
    out_all = traits.Bool(argstr='--Oall', desc='output everything')
    out_unmix = traits.Bool(argstr='--Ounmix', desc='output unmixing matrix')
    out_stats = traits.Bool(argstr='--Ostats', desc='output thresholded maps and probability maps')
    out_pca = traits.Bool(argstr='--Opca', desc='output PCA results')
    out_white = traits.Bool(argstr='--Owhite', desc='output whitening/dewhitening matrices')
    out_orig = traits.Bool(argstr='--Oorig', desc='output the original ICs')
    out_mean = traits.Bool(argstr='--Omean', desc='output mean volume')
    report_maps = traits.Str(argstr='--report_maps=%s', desc='control string for spatial map images (see slicer)')
    remove_deriv = traits.Bool(argstr='--remove_deriv', desc='removes every second entry in paradigm file (EV derivatives)')