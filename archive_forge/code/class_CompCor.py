import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
class CompCor(SimpleInterface):
    """
    Interface with core CompCor computation, used in aCompCor and tCompCor.

    CompCor provides three pre-filter options, all of which include per-voxel
    mean removal:

      - ``'polynomial'``: Legendre polynomial basis
      - ``'cosine'``: Discrete cosine basis
      - ``False``: mean-removal only

    In the case of ``polynomial`` and ``cosine`` filters, a pre-filter file may
    be saved with a row for each volume/timepoint, and a column for each
    non-constant regressor.
    If no non-constant (mean-removal) columns are used, this file may be empty.

    If ``ignore_initial_volumes`` is set, then the specified number of initial
    volumes are excluded both from pre-filtering and CompCor component
    extraction.
    Each column in the components and pre-filter files are prefixe with zeros
    for each excluded volume so that the number of rows continues to match the
    number of volumes in the input file.
    In addition, for each excluded volume, a column is added to the pre-filter
    file with a 1 in the corresponding row.

    Example
    -------
    >>> ccinterface = CompCor()
    >>> ccinterface.inputs.realigned_file = 'functional.nii'
    >>> ccinterface.inputs.mask_files = 'mask.nii'
    >>> ccinterface.inputs.num_components = 1
    >>> ccinterface.inputs.pre_filter = 'polynomial'
    >>> ccinterface.inputs.regress_poly_degree = 2

    """
    input_spec = CompCorInputSpec
    output_spec = CompCorOutputSpec
    _references = [{'tags': ['method', 'implementation'], 'entry': BibTeX('@article{compcor_2007,\n    title = {A component based noise correction method (CompCor) for BOLD and perfusion based},\n    volume = {37},\n    number = {1},\n    doi = {10.1016/j.neuroimage.2007.04.042},\n    urldate = {2016-08-13},\n    journal = {NeuroImage},\n    author = {Behzadi, Yashar and Restom, Khaled and Liau, Joy and Liu, Thomas T.},\n    year = {2007},\n    pages = {90-101}\n}')}]

    def __init__(self, *args, **kwargs):
        """exactly the same as compcor except the header"""
        super(CompCor, self).__init__(*args, **kwargs)
        self._header = 'CompCor'

    def _run_interface(self, runtime):
        mask_images = []
        if isdefined(self.inputs.mask_files):
            mask_images = combine_mask_files(self.inputs.mask_files, self.inputs.merge_method, self.inputs.mask_index)
        if self.inputs.use_regress_poly:
            self.inputs.pre_filter = 'polynomial'
        degree = self.inputs.regress_poly_degree if self.inputs.pre_filter == 'polynomial' else 0
        imgseries = nb.load(self.inputs.realigned_file)
        if len(imgseries.shape) != 4:
            raise ValueError('{} expected a 4-D nifti file. Input {} has {} dimensions (shape {})'.format(self._header, self.inputs.realigned_file, len(imgseries.shape), imgseries.shape))
        if len(mask_images) == 0:
            img = nb.Nifti1Image(np.ones(imgseries.shape[:3], dtype=bool), affine=imgseries.affine, header=imgseries.header)
            mask_images = [img]
        skip_vols = self.inputs.ignore_initial_volumes
        if skip_vols:
            imgseries = imgseries.__class__(imgseries.dataobj[..., skip_vols:], imgseries.affine, imgseries.header)
        mask_images = self._process_masks(mask_images, imgseries.dataobj)
        TR = 0
        if self.inputs.pre_filter == 'cosine':
            if isdefined(self.inputs.repetition_time):
                TR = self.inputs.repetition_time
            else:
                try:
                    TR = imgseries.header.get_zooms()[3]
                    if imgseries.header.get_xyzt_units()[1] == 'msec':
                        TR /= 1000
                except (AttributeError, IndexError):
                    TR = 0
                if TR == 0:
                    raise ValueError('{} cannot detect repetition time from image - Set the repetition_time input'.format(self._header))
        if isdefined(self.inputs.variance_threshold):
            components_criterion = self.inputs.variance_threshold
        elif isdefined(self.inputs.num_components):
            components_criterion = self.inputs.num_components
        else:
            components_criterion = 6
            IFLOGGER.warning('`num_components` and `variance_threshold` are not defined. Setting number of components to 6 for backward compatibility. Please set either `num_components` or `variance_threshold`, as this feature may be deprecated in the future.')
        components, filter_basis, metadata = compute_noise_components(imgseries.get_fdata(dtype=np.float32), mask_images, components_criterion, self.inputs.pre_filter, degree, self.inputs.high_pass_cutoff, TR, self.inputs.failure_mode, self.inputs.mask_names)
        if skip_vols:
            old_comp = components
            nrows = skip_vols + components.shape[0]
            components = np.zeros((nrows, components.shape[1]), dtype=components.dtype)
            components[skip_vols:] = old_comp
        components_file = os.path.join(os.getcwd(), self.inputs.components_file)
        components_header = self._make_headers(components.shape[1])
        np.savetxt(components_file, components, fmt=b'%.10f', delimiter='\t', header='\t'.join(components_header), comments='')
        self._results['components_file'] = os.path.join(runtime.cwd, self.inputs.components_file)
        save_pre_filter = False
        if self.inputs.pre_filter in ['polynomial', 'cosine']:
            save_pre_filter = self.inputs.save_pre_filter
        if save_pre_filter:
            self._results['pre_filter_file'] = save_pre_filter
            if save_pre_filter is True:
                self._results['pre_filter_file'] = os.path.join(runtime.cwd, 'pre_filter.tsv')
            ftype = {'polynomial': 'Legendre', 'cosine': 'Cosine'}[self.inputs.pre_filter]
            ncols = filter_basis.shape[1] if filter_basis.size > 0 else 0
            header = ['{}{:02d}'.format(ftype, i) for i in range(ncols)]
            if skip_vols:
                old_basis = filter_basis
                filter_basis = np.zeros((nrows, ncols + skip_vols), dtype=filter_basis.dtype)
                if old_basis.size > 0:
                    filter_basis[skip_vols:, :ncols] = old_basis
                filter_basis[:skip_vols, -skip_vols:] = np.eye(skip_vols)
                header.extend(['NonSteadyStateOutlier{:02d}'.format(i) for i in range(skip_vols)])
            np.savetxt(self._results['pre_filter_file'], filter_basis, fmt=b'%.10f', delimiter='\t', header='\t'.join(header), comments='')
        metadata_file = self.inputs.save_metadata
        if metadata_file:
            self._results['metadata_file'] = metadata_file
            if metadata_file is True:
                self._results['metadata_file'] = os.path.join(runtime.cwd, 'component_metadata.tsv')
            components_names = np.empty(len(metadata['mask']), dtype='object_')
            retained = np.where(metadata['retained'])
            not_retained = np.where(np.logical_not(metadata['retained']))
            components_names[retained] = components_header
            components_names[not_retained] = ['dropped{}'.format(i) for i in range(len(not_retained[0]))]
            with open(self._results['metadata_file'], 'w') as f:
                f.write('\t'.join(['component'] + list(metadata.keys())) + '\n')
                for i in zip(components_names, *metadata.values()):
                    f.write('{0[0]}\t{0[1]}\t{0[2]:.10f}\t{0[3]:.10f}\t{0[4]:.10f}\t{0[5]}\n'.format(i))
        return runtime

    def _process_masks(self, mask_images, timeseries=None):
        return mask_images

    def _make_headers(self, num_col):
        header = self.inputs.header_prefix if isdefined(self.inputs.header_prefix) else self._header
        headers = ['{}{:02d}'.format(header, i) for i in range(num_col)]
        return headers