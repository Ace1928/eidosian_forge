import os
import numpy as np
import nibabel as nb
from ..interfaces.base import (
class SignalExtraction(NilearnBaseInterface, SimpleInterface):
    """
    Extracts signals over tissue classes or brain regions

    >>> seinterface = SignalExtraction()
    >>> seinterface.inputs.in_file = 'functional.nii'
    >>> seinterface.inputs.label_files = 'segmentation0.nii.gz'
    >>> seinterface.inputs.out_file = 'means.tsv'
    >>> segments = ['CSF', 'GrayMatter', 'WhiteMatter']
    >>> seinterface.inputs.class_labels = segments
    >>> seinterface.inputs.detrend = True
    >>> seinterface.inputs.include_global = True
    """
    input_spec = SignalExtractionInputSpec
    output_spec = SignalExtractionOutputSpec

    def _run_interface(self, runtime):
        maskers = self._process_inputs()
        signals = []
        for masker in maskers:
            signals.append(masker.fit_transform(self.inputs.in_file))
        region_signals = np.hstack(signals)
        output = np.vstack((self.inputs.class_labels, region_signals.astype(str)))
        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        np.savetxt(self._results['out_file'], output, fmt=b'%s', delimiter='\t')
        return runtime

    def _process_inputs(self):
        """validate and  process inputs into useful form.
        Returns a list of nilearn maskers and the list of corresponding label
        names."""
        import nilearn.input_data as nl
        import nilearn.image as nli
        label_data = nli.concat_imgs(self.inputs.label_files)
        maskers = []
        if np.amax(label_data.dataobj) > 1:
            n_labels = np.amax(label_data.dataobj)
            maskers.append(nl.NiftiLabelsMasker(label_data))
        else:
            n_labels = label_data.shape[3]
            if self.inputs.incl_shared_variance:
                for img in nli.iter_img(label_data):
                    maskers.append(nl.NiftiMapsMasker(self._4d(img.dataobj, img.affine)))
            else:
                maskers.append(nl.NiftiMapsMasker(label_data))
        if not np.isclose(int(n_labels), n_labels):
            raise ValueError('The label files {} contain invalid value {}. Check input.'.format(self.inputs.label_files, n_labels))
        if len(self.inputs.class_labels) != n_labels:
            raise ValueError('The length of class_labels {} does not match the number of regions {} found in label_files {}'.format(self.inputs.class_labels, n_labels, self.inputs.label_files))
        if self.inputs.include_global:
            global_label_data = label_data.dataobj.sum(axis=3)
            global_label_data = np.rint(global_label_data).clip(0, 1).astype('u1')
            global_label_data = self._4d(global_label_data, label_data.affine)
            global_masker = nl.NiftiLabelsMasker(global_label_data, detrend=self.inputs.detrend)
            maskers.insert(0, global_masker)
            self.inputs.class_labels.insert(0, 'GlobalSignal')
        for masker in maskers:
            masker.set_params(detrend=self.inputs.detrend)
        return maskers

    def _4d(self, array, affine):
        """takes a 3-dimensional numpy array and an affine,
        returns the equivalent 4th dimensional nifti file"""
        return nb.Nifti1Image(array[:, :, :, np.newaxis], affine)